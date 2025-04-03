import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import math
from Datasets.dataset_gaussian import GaussianMixtureDataset  # Removed PalindromeDataset import

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Params
seq_len = 20
batch_size = 64
hidden_size = 128
vocab_size = 10

class PositionalEncoding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 20):
        super().__init__()
        self.d_model = d_model
        # Projection layer to map one-hot to embedding space
        self.token_proj = nn.Linear(vocab_size, d_model)
        
        # Positional embedding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x_onehot):
        seq_len, batch_size, vocab_size = x_onehot.shape
        emb = self.token_proj(x_onehot)
        emb = emb + self.pe[:seq_len]
        return emb

class AutoregressiveTransformer(pl.LightningModule):
    def __init__(self, vocab_size=10, d_model=64, nhead=2, d_hid=128,  # Changed vocab_size to 10
                 num_layers=2, dropout=0.1, max_len=20, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.lr = lr
        
        eye = torch.eye(vocab_size, dtype=torch.float32)
        self.register_buffer("one_hot_encoding", eye)

        # Positional embedding
        self.pos_encoder = PositionalEncoding(vocab_size, d_model, max_len)

        # Decoder only
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_hid,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output_linear = nn.Linear(d_model, vocab_size)
        self.criterion = nn.CrossEntropyLoss()
    
    def causal_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask.to(self.device)
    
    def forward(self, x_tokens):
        seq_len, batch_size = x_tokens.shape
        
        x_1h = self.one_hot_encoding[x_tokens]
        emb = self.pos_encoder(x_1h)

        # Build causal mask
        tgt_mask = self.causal_mask(seq_len)

        mem_len = 1
        memory = torch.zeros(mem_len, batch_size, self.d_model, device=self.device)

        out = self.transformer_decoder(
            tgt=emb,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=None
        )
        
        logits = self.output_linear(out)
        return logits

    def training_step(self, batch, batch_idx):
        batch = batch.to(self.device).transpose(0, 1)
        input_seq = batch[:-1]
        target_seq = batch[1:]
        
        logits = self(input_seq)
        logits_2d = logits.view(-1, self.vocab_size)
        targets_1d = target_seq.reshape(-1)
        
        loss = self.criterion(logits_2d, targets_1d)
        
        # Calculate accuracy
        preds = torch.argmax(logits_2d, dim=-1)
        accuracy = (preds == targets_1d).float().mean()
        
        self.log('train_loss', loss)
        self.log('train_acc', accuracy)
        
        return loss

    def validation_step(self, batch, batch_idx):
        batch = batch.to(self.device).transpose(0, 1)
        input_seq = batch[:-1]
        target_seq = batch[1:]
        
        logits = self(input_seq)
        logits_2d = logits.view(-1, self.vocab_size)
        targets_1d = target_seq.reshape(-1)
        
        loss = self.criterion(logits_2d, targets_1d)
        
        # Calculate accuracy
        preds = torch.argmax(logits_2d, dim=-1)
        accuracy = (preds == targets_1d).float().mean()
        
        self.log('val_loss', loss)
        self.log('val_acc', accuracy)
        
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)

    def generate(self, batch_size=1, max_length=20, temperature=1.0):
        self.eval()
        with torch.no_grad():
            init_tokens = torch.randint(0, self.vocab_size,
                                      (1, batch_size), device=self.device)
            generated_seq = init_tokens
            
            for _ in range(max_length - 1):
                logits = self(generated_seq)
                last_logits = logits[-1, :, :]
                
                if temperature != 1.0:
                    last_logits = last_logits / temperature
                
                probs = torch.softmax(last_logits, dim=-1)
                next_tokens = torch.multinomial(probs, 1).squeeze(-1)
                
                next_tokens = next_tokens.unsqueeze(0)
                generated_seq = torch.cat([generated_seq, next_tokens], dim=0)
            
            return generated_seq

class DataModule(pl.LightningDataModule):
    def __init__(self, dataset_class, seq_len, batch_size=32):
        super().__init__()
        self.dataset_class = dataset_class
        self.seq_len = seq_len
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = self.dataset_class(self.seq_len)
        self.val_dataset = self.dataset_class(self.seq_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

# Training setup
model = AutoregressiveTransformer(
    vocab_size=10,  # Changed vocab_size to 10
    d_model=64,
    nhead=2,
    d_hid=128,
    num_layers=2,
    dropout=0.1,
    max_len=seq_len,
    lr=1e-4
)

# Create trainer for each dataset
datasets = {
    'gaussian': GaussianMixtureDataset
}

import matplotlib.pyplot as plt

for dataset_name, dataset_class in datasets.items():
    print(f"\nTraining on {dataset_name} dataset:")
    datamodule = DataModule(dataset_class, seq_len=seq_len, batch_size=batch_size)
    
    trainer = pl.Trainer(
        max_epochs=100,  # Changed from 1 to 10 for more iterations
        accelerator='auto',
        devices=1,
        enable_progress_bar=True,
    )
    
    trainer.fit(model, datamodule)
    
    # Generate examples
    print(f"\n=== Generation Examples for {dataset_name} ===")
    model.eval()
    for temp in [0.8, 1.0, 1.2]:
        print(f"\nTemperature: {temp}")
        generated = model.generate(batch_size, max_length=seq_len, temperature=temp)
        generated = generated.transpose(0,1).cpu().tolist()
        for i, seq in enumerate(generated):
            print(f"  Sample {i+1}: {seq}")

    # Graph the distribution of training samples
    training_samples = datamodule.train_dataset.samples
    plt.figure(figsize=(12, 6))
    plt.hist(training_samples, bins=10, alpha=0.5, label='Training Samples', density=True)

    # Graph the distribution of generated samples
    generated_flat = [item for sublist in generated for item in sublist]
    plt.hist(generated_flat, bins=10, alpha=0.5, label='Generated Samples', density=True)

    plt.title(f'Distribution of Training and Generated Samples for {dataset_name}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

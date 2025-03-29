import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Params 
seq_len = 20
batch_size = 32
hidden_size = 128
vocab_size = 2

class PalindromeDataset(Dataset):
    def __init__(self, seq_len=20):
        self.seq_len = seq_len
        
    def __len__(self):
        return 100000  # effectively infinite

    def __getitem__(self, idx):
        half = np.random.randint(0, 2, size=self.seq_len // 2)
        palindrome = np.concatenate([half, half[::-1]])
        return torch.tensor(palindrome, dtype=torch.long)

def is_palindrome(seq):
    return seq == seq[::-1]


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

class AutoregressiveTransformer(nn.Module):
    def __init__(self, vocab_size=2, d_model=64, nhead=2, d_hid=128,
                 num_layers=2, dropout=0.1, max_len=20):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        
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
    
    def causal_mask(self, size):
        # Upper triangular mask 
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask.to(device)
    
    def forward(self, x_tokens):
        seq_len, batch_size = x_tokens.shape
        
        x_1h = self.one_hot_encoding[x_tokens]   
        emb = self.pos_encoder(x_1h)             

        # Build causal mask
        tgt_mask = self._causal_mask(seq_len)    # [seq_len, seq_len]

        mem_len = 1
        memory = torch.zeros(mem_len, batch_size, self.d_model, device=device)

        out = self.transformer_decoder(
            tgt=emb,
            memory=memory,            # all zeros
            tgt_mask=tgt_mask,
            memory_mask=None          
        )
        
        logits = self.output_linear(out)          
        return logits
    
    def generate(self, batch_size=1, max_length=20, temperature=1.0):
        # Autoregressively sample one token a time
        self.eval()
        with torch.no_grad():
            # Start from random
            init_tokens = torch.randint(0, self.vocab_size,
                                        (1, batch_size), device=device) 
            generated_seq = init_tokens
            
            for _ in range(max_length - 1):
                # Forward pass 
                logits = self(generated_seq)              
                last_logits = logits[-1, :, :]             
                
                # Apply temperature
                if temperature != 1.0:
                    last_logits = last_logits / temperature
                
                # Sample
                probs = torch.softmax(last_logits, dim=-1) 
                next_tokens = torch.multinomial(probs, 1).squeeze(-1)  
                
                # Append
                next_tokens = next_tokens.unsqueeze(0)    
                generated_seq = torch.cat([generated_seq, next_tokens], dim=0)
            
            return generated_seq

train_dataset = PalindromeDataset(seq_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_iter = iter(train_loader)

model = AutoregressiveTransformer(
    vocab_size=2,
    d_model=64,
    nhead=2,
    d_hid=128,
    num_layers=2,
    dropout=0.1,
    max_len=seq_len
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

num_iter = 5000
eval_step = 100
generate_step = 1000
iteration = 0

while iteration < num_iter:
    iteration += 1
    model.train()
    total_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        batch = next(train_iter)

    batch = batch.to(device).transpose(0, 1)
    
    input_seq = batch[:-1]   
    target_seq = batch[1:]

    optimizer.zero_grad()
    logits = model(input_seq)  
    
    # Flatten 
    logits_2d = logits.view(-1, model.vocab_size)     
    targets_1d = target_seq.reshape(-1)              

    loss = criterion(logits_2d, targets_1d)
    loss.backward()
    optimizer.step()
    
    # Accuracy
    preds = torch.argmax(logits_2d, dim=-1)
    correct_preds += (preds == targets_1d).sum().item()
    total_preds += targets_1d.size(0)
    total_loss += loss.item()
    
    if iteration % eval_step == 0:
        avg_loss = total_loss / eval_step
        accuracy = correct_preds / total_preds
        print(f"Iteration {iteration}, Loss={avg_loss:.4f}, Accuracy={accuracy*100:.1f}%")
        total_loss = 0.0
        correct_preds = 0
        total_preds = 0

    # Periodically generate
    if iteration % generate_step == 0:
        model.eval()
        print("\n=== Generation Examples ===")
        for temp in [0.8, 1.0, 1.2]:
            print(f"\nTemperature: {temp}")
            generated = model.generate(batch_size=5, max_length=seq_len, temperature=temp)
            # => [seq_len, 5]
            generated = generated.transpose(0,1).cpu().tolist()  # => [5, seq_len]
            for i, seq in enumerate(generated):
                mark = "✓" if is_palindrome(seq) else "✗"
                print(f"  Sample {i+1}: {seq} {mark}")

# Final evaluation
print("\n=== Final Generation Results ===")
model.eval()
for temp in [0.7, 0.8, 0.9, 1.0]:
    total_samples = 20
    valid_count = 0
    generated = model.generate(batch_size=total_samples, max_length=seq_len, temperature=temp)
    generated = generated.transpose(0,1).cpu().tolist()
    for seq in generated:
        if is_palindrome(seq):
            valid_count += 1
    print(f"Temperature={temp}, Palindromes={valid_count}/{total_samples} ({valid_count/total_samples*100:.1f}%)")

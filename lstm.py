import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Params 
seq_len = 20
batch_size = 32
hidden_size = 128
num_layers = 1
num_iter = 2000
eval_step = 100
vocab_size = 2

class PalindromeDataset(Dataset):
    def __init__(self, seq_length=20):
        self.seq_length = seq_length
        
    def __len__(self):
        return 100000  # effectively infinite

    def __getitem__(self, idx):
        half = np.random.randint(0, 2, size=self.seq_length // 2)
        palindrome = np.concatenate((half, half[::-1]))
        return torch.tensor(palindrome, dtype=torch.long)

def is_palindrome(seq):
    return seq == seq[::-1]

# Create dataset and DataLoader
train_dataset = PalindromeDataset(seq_length=seq_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_loader_iter = iter(train_loader)

class LSTMPalindromeModel(nn.Module):
    def __init__(self, vocab_size=2, hidden_size=128, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # LSTM
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=False)
        
        self.output_fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        
        out, hidden = self.lstm(emb, hidden)
        
        logits = self.output_fc(out)
        return logits, hidden
    
    def generate(self, batch_size=1, max_length=20, temperature=1.0):
        self.eval()
        with torch.no_grad():
            hidden = None
            
            init_tokens = torch.randint(0, self.vocab_size, (batch_size,), device=device)
            generated_seq = init_tokens.unsqueeze(0)  # [1, batch_size]
            
            for _ in range(max_length - 1):
                # Forward pass 
                logits, hidden = self(generated_seq[-1:].to(device), hidden)
                
                last_logits = logits[-1, :, :]
                
                if temperature != 1.0:
                    last_logits = last_logits / temperature
                
                # Sample next 
                probs = torch.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).squeeze(-1)
                
                next_token = next_token.unsqueeze(0)
                generated_seq = torch.cat([generated_seq, next_token], dim=0)
            
            return generated_seq


model = LSTMPalindromeModel(vocab_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_iter = 10000
eval_step = 100
generate_step = 500

iteration = 0
train_loader_iter = iter(train_loader)

while iteration < num_iter:
    iteration += 1
    model.train()
    total_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    try:
        batch = next(train_loader_iter)
    except StopIteration:
        train_loader_iter = iter(train_loader)
        batch = next(train_loader_iter)
    
    batch = batch.transpose(0, 1).to(device)
    
    input_seq = batch[:-1]   
    target_seq = batch[1:]   
    
    optimizer.zero_grad()
    
    # Forward pass
    logits, _ = model(input_seq)
    
    # Flatten
    logits_2d = logits.view(-1, vocab_size)      
    targets_1d = target_seq.reshape(-1)            
    
    loss = criterion(logits_2d, targets_1d)
    
    # Accuracy
    preds = torch.argmax(logits_2d, dim=-1)
    correct_preds += (preds == targets_1d).sum().item()
    total_preds += targets_1d.size(0)
    
    # Backprop
    loss.backward()
    optimizer.step()
    
    total_loss += loss.item()
    
    avg_loss = total_loss / eval_step
    accuracy = correct_preds / total_preds if total_preds > 0 else 0

    if iteration % eval_step == 0:
        print(f"Iteration {iteration}, Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        total_loss = 0.0
        correct_preds = 0
        total_preds = 0
    
    # Generate samples every generate_step iterations
    if iteration % generate_step == 0:
        model.eval()
        print("\n=== Generation Examples ===")
        for temp in [0.8, 1.0, 1.2]:
            print(f"\nTemperature: {temp}")
            generated = model.generate(batch_size=5, max_length=seq_len, temperature=temp)
            generated = generated.transpose(0, 1).cpu().tolist()

            for i, seq in enumerate(generated):
                is_valid = is_palindrome(seq)
                print(f"  Sample {i+1}: {seq} {'✓' if is_valid else '✗'}")

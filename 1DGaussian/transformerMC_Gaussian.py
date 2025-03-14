import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader
import random
import math
from torch.nn import TransformerDecoder, TransformerDecoderLayer
#%%
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
seq_len = 20
batch_size = 32
d_model = 128  # Increased model dimension
vocab_size = 12 # 2 for Binary data 10 for floating point data
MASK_TOKEN = 12  # Using 2 as the mask token ID (beyond 0 and 1 for binary data)
#%%
# Generate palindrome data

# Custom dataset for continuously generating palindromes
class PalindromeDataset(Dataset):
    def __init__(self, seq_length=20):
        self.seq_length = seq_length
        
    def __len__(self):
        # Just return a large number, we'll generate data on the fly
        return 100000  # Effectively infinite
        
    def __getitem__(self, idx):
        # Generate a new palindrome for each request
        half = np.random.randint(0, 2, size=self.seq_length // 2)
        palindrome = np.concatenate((half, half[::-1]))
        return torch.tensor(palindrome, dtype=torch.long)



class GaussianMixtureDataset(Dataset):
    def __init__(self, num_samples=10000000, means=[0, 5], stds=[1, 1], weights=[0.5, 0.5], seq_length = 20):
        """
        Args:
            num_samples (int): Number of samples to generate.
            means (list): Means of the Gaussian components.
            stds (list): Standard deviations of the Gaussian components.
            weights (list): Mixture weights for each component.
        """
        self.num_samples = num_samples
        self.means = means
        self.stds = stds
        self.weights = weights
        
        # Generate samples from the Gaussian mixture
        self.samples = self._generate_samples()
    
    def _generate_samples(self):
        """Generate samples from the Gaussian Mixture."""
        components = np.random.choice(len(self.means), size=self.num_samples, p=self.weights)
        samples = np.array([
            np.random.normal(self.means[c], self.stds[c]) for c in components
        ])
        return samples
    
    def _discretize_sample(self, sample):
        """Convert a float sample into a list of its digits."""
        sample_str = f"{sample:.19f}"  # Keep four decimal places
        return [int(char) for char in sample_str if char.isdigit()]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return torch.tensor(self._discretize_sample(sample), dtype=torch.long)



def is_palindrome(seq):
    return seq == seq[::-1]



# Create dataset and dataloader
#train_dataset = PalindromeDataset(seq_length=seq_len)
#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

train_dataset=GaussianMixtureDataset()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# Positional encoding

class PositionalEncoding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 20):
        super().__init__()
        # Projection layer to map one-hot to embedding space
        self.token_proj = nn.Linear(vocab_size, d_model)
        
        # Positional embedding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project one-hot to embedding space
        x = self.token_proj(x)
        
        # Add positional encoding
        x = x + self.pe[:x.size(0)]
        return x
#%%
# Masked Token Transformer
class MaskedTokenTransformer(nn.Module):
    def __init__(self, vocab_size=11, mask_token=12, d_model=128, nhead=4, 
                 d_hid=512, num_layers=3, dropout=0.2, max_len=20):
        super(MaskedTokenTransformer, self).__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.mask_token = mask_token
        
        # Embedding layer with mask token (+1 for the mask token)
          # Create a fixed one-hot encoding matrix
        self.register_buffer('one_hot_encoding', torch.eye(vocab_size + 1, vocab_size + 1, dtype=torch.float32))
        
        self.pos_encoder = PositionalEncoding(vocab_size+1, d_model, max_len)
        
        # Transformer layers
        decoder_layers = TransformerDecoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_layers)
        
        # Output projection
        self.linear = nn.Linear(d_model, vocab_size)  # Only project to actual vocab (0,1), not mask
    
    def generate_mask(self, size):
        # We don't need a causal mask here as we're processing all tokens
       
        return None
    
    def forward(self, x, attention_mask=None):
        """
        x: Input tensor with possible mask tokens (seq_len, batch_size)
        attention_mask: Optional mask to prevent attending to certain positions
        """
        # Embedding and positional encoding
        x = self.one_hot_encoding[x]
        x = self.pos_encoder(x)
        
        # Transformer processing
        # We use the input as both query and key/value, without a separate memory
        x = self.transformer_decoder(x, x, tgt_mask=self.generate_mask(x.size(0)),
                                     tgt_key_padding_mask=attention_mask)
        
        # Project to vocabulary (excluding mask token)
        logits = self.linear(x)
        
        return logits
    
    def create_masked_inputs(self, batch, mask_prob=0.8):
        """
        Creates masked versions of the input sequences for training.
        Returns:
        - masked_batch: Input with some tokens masked
        - target_tensor: Original values of the masked positions
        - mask_tensor: Boolean tensor indicating which positions were masked
        """
        # Create a copy of the input batch
        masked_batch = batch.clone()
        
        # Create a mask tensor (True where tokens will be masked)
        # Exclude special tokens from masking (not applicable for this binary case)
        mask_tensor = torch.rand_like(batch.float()) < mask_prob
        
        # Replace masked positions with mask token
        masked_batch[mask_tensor] = self.mask_token
        
        # Target is the original batch
        target_tensor = batch.clone()
        
        return masked_batch, target_tensor, mask_tensor
    
    def generate(self, batch_size=1, generation_method='iterative', max_iterations=20, temperature=1.0):
        """
        Generate sequences by progressively unmasking tokens.
        
        generation_method:
        - 'iterative': Unmask one token at a time
        - 'parallel': Predict all tokens in parallel (like BERT)
        - 'outside_in': Unmask from outside to inside (useful for palindromes)
        """
        self.eval()
        with torch.no_grad():
            # Start with all tokens masked
            masked_seq = torch.full((self.max_len, batch_size), self.mask_token, 
                                  dtype=torch.long, device=device)
            
            if generation_method == 'iterative':
                # Initialize mask tracking (all positions start masked)
                masked_positions = torch.ones((self.max_len, batch_size), dtype=torch.bool, device=device)
                
                # Iteratively unmask one position at a time
                for _ in range(min(self.max_len, max_iterations)):
                    # Get model predictions for current masked sequence
                    logits = self(masked_seq)
                    
                    # Apply temperature scaling
                    if temperature != 1.0:
                        logits = logits / temperature
                    
                    probs = torch.softmax(logits, dim=-1)
                    
                    # Find position with highest confidence to unmask
                    # We take max probability across vocabulary dimension
                    max_probs, _ = probs.max(dim=-1)  # (seq_len, batch_size)
                    
                    # Only consider still-masked positions
                    max_probs = max_probs * masked_positions.float()
                    
                    # For each sequence in batch, find highest confidence position
                    for b in range(batch_size):
                        # Skip if no masked positions left
                        if not masked_positions[:, b].any():
                            continue
                            
                        # Find position with highest confidence
                        pos = max_probs[:, b].argmax().item()
                        
                        # Get prediction for this position
                        token_probs = probs[pos, b]
                        token = torch.multinomial(token_probs, 1).item()
                        
                        # Unmask this position
                        masked_seq[pos, b] = token
                        masked_positions[pos, b] = False
                
                return masked_seq
            
            elif generation_method == 'outside_in':
                # Special method for palindromes - unmask from outside to inside
                middle = self.max_len // 2
                
                # Create unmasking order (from outside to inside)
                unmask_order = []
                for i in range(middle):
                    unmask_order.append(i)  # Left side
                    unmask_order.append(self.max_len - 1 - i)  # Right side
                
                # Add middle position if sequence length is odd
                if self.max_len % 2 != 0:
                    unmask_order.append(middle)
                
                # Track masked positions
                masked_positions = torch.ones((self.max_len, batch_size), dtype=torch.bool, device=device)
                
                # Unmask tokens according to order
                for pos in unmask_order:
                    # Get model predictions for current masked sequence
                    logits = self(masked_seq)
                    
                    # Apply temperature scaling
                    if temperature != 1.0:
                        logits = logits / temperature
                    
                    # Get token predictions
                    probs = torch.softmax(logits[pos], dim=-1)
                    tokens = torch.multinomial(probs, 1).squeeze(-1)
                    
                    # Unmask this position
                    masked_seq[pos] = tokens
                    masked_positions[pos] = False
                    
                    # For palindromes, enforce symmetry after predicting first half
                    if pos < middle and self.max_len - 1 - pos in unmask_order:
                        # Copy symmetric token for the second half
                        masked_seq[self.max_len - 1 - pos] = tokens
                        masked_positions[self.max_len - 1 - pos] = False
                        
                        # Remove the symmetric position from unmasking order
                        try:
                            unmask_order.remove(self.max_len - 1 - pos)
                        except ValueError:
                            pass
                
                return masked_seq
            
            else:
                raise ValueError(f"Unknown generation method: {generation_method}")
# Create the model and move to device
model = MaskedTokenTransformer(
    vocab_size=vocab_size,
    mask_token=MASK_TOKEN,
    d_model=d_model,
    nhead=4,
    d_hid=64,
    num_layers=4,
    dropout=0.2,
    max_len=seq_len
).to(device)




# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)


#%%
# Training loop
num_iter = 100
mask_probability = 0.2  # Start with low
iteration=0
while iteration < num_iter:
    iteration+=1
    model.train()
    total_loss = 0
    correct_preds = 0
    total_preds = 0
    
    
    try:
        batch = next(train_loader_iter)
    except Exception as e:
        print(e)
        # Recreate the iterator when it's exhausted
        train_loader_iter = iter(train_loader)
        batch = next(train_loader_iter)
  
    #Process batch
    batch = batch.long().to(device)  # Shape: (batch_size, seq_len)
    batch = batch.transpose(0, 1)  # Shape: (seq_len, batch_size)
    
    optimizer.zero_grad()
    
    # Create masked batch
    masked_batch, target_batch, mask_tensor = model.create_masked_inputs(batch, mask_prob=mask_probability)
    
    # Forward pass
    logits = model(masked_batch)
    
    
    # Reshape logits to [batch_size*seq_len, vocab_size]
    reshaped_logits = logits.permute(1, 0, 2).reshape(-1, vocab_size)

    # Reshape targets and mask to [batch_size*seq_len]
    reshaped_targets = target_batch.permute(1, 0).reshape(-1)
    reshaped_mask = mask_tensor.permute(1, 0).reshape(-1)

    # Select only the masked positions
    logits_masked = reshaped_logits[reshaped_mask]  # Shape: [num_masked, vocab_size]
    targets_masked = reshaped_targets[reshaped_mask]  # Shape: [num_masked]

    
    # Compute loss
    loss = criterion(logits_masked, targets_masked)
    
    # Compute accuracy
    preds = torch.argmax(logits_masked, dim=-1)
    correct_preds += (preds == targets_masked).sum().item()
    total_preds += targets_masked.size(0)
    
    # Backward pass
    loss.backward()
    
    # Update weights
    optimizer.step()
    
    total_loss += loss.item()
    
    #if batch_idx % 10 == 0:
        #print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}, Mask prob: {mask_probability:.2f}")
    

model.eval()
print("\n=== Generation Examples ===")

generated_samples = model.generate(batch_size=1000, generation_method='iterative', temperature=1.0)
generated_samples = generated_samples.transpose(0, 1).cpu().tolist()  # Convert to batch-first

# Convert discrete samples back to float values
def decode_discrete_to_float(seq):
    """Converts a list of digits back to a floating-point number."""
    if len(seq) < 2:  # Ensure there's at least one digit before the decimal
        return 0.0
    
    first_digit = str(seq[0])  # First digit remains before the decimal
    decimal_part = "".join(map(str, seq[1:]))  # Rest forms the decimal part
    reconstructed = float(f"{first_digit}.{decimal_part}")  # Recreate float

    return reconstructed

generated_floats = np.array([decode_discrete_to_float(seq) for seq in generated_samples])

# Load Gaussian Mixture Samples (assuming dataset object exists)
original_floats = np.array(train_dataset.samples[:1000])  # Sample from original dataset

# Plot histograms to compare
plt.figure(figsize=(10, 5))
plt.hist(original_floats, bins=50, alpha=0.5, label="Original Gaussian Mixture")
plt.hist(generated_floats, bins=50, alpha=0.5, label="Generated Samples")
plt.legend()
plt.title("Comparison of Gaussian Mixture vs. Generated Samples")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# Compute mean and variance for comparison
print(f"Original Mean: {np.mean(original_floats):.4f}, Variance: {np.var(original_floats):.4f}")
print(f"Generated Mean: {np.mean(generated_floats):.4f}, Variance: {np.var(generated_floats):.4f}")


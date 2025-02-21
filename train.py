import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from binary_dataset import BinarySequenceDataset
from model import DiscreteDiffusionSimpleTransformer as DiscreteDiffusionTransformer

# Hyperparameters
MODEL_DIM = 16 
NUM_HEADS = 8 
NUM_LAYERS = 6  
SEQ_LENGTH = 16 
VOCAB_SIZE = 10  # Character distribution size (modify if needed)
BATCH_SIZE = 32  
EPOCHS = 10  
LR = 0.001  
delta = [MODEL_DIM, NUM_HEADS, NUM_LAYERS, SEQ_LENGTH, VOCAB_SIZE, BATCH_SIZE, EPOCHS, LR]
delta=[str(i) for i in delta]
PATH_TO_MODEL = "Models/simple_model/"

# Load dataset
dataset = BinarySequenceDataset(num_samples=1000, word_size=8)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model, loss, optimizer
model = DiscreteDiffusionTransformer(MODEL_DIM, NUM_HEADS, NUM_LAYERS, VOCAB_SIZE, SEQ_LENGTH)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in dataloader:
        src, _ = batch  # Fix: Only use the first element, which is the tensor
        src = src.long()  # Ensure it's of type long for embedding lookup

        optimizer.zero_grad()
        output = model(src)  # Shape: [batch, seq_length, vocab_size]
        
        loss = criterion(output.view(-1, VOCAB_SIZE), src.view(-1))  # Predict distribution
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

torch.save(model.state_dict(), f"{PATH_TO_MODEL}/transformer_diffusion_{'_'.join(delta)}.pth")
print("Model saved.")


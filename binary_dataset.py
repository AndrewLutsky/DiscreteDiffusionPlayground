import torch
from torch.utils.data import Dataset, DataLoader
import random

class BinarySequenceDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sequence = [random.randint(0, 1) for _ in range(20)]  # Generate random 0s and 1s
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32)  # Convert to tensor
        sequence_str = ''.join(map(str, sequence))  # Convert to string
        
        return sequence_tensor, sequence_str



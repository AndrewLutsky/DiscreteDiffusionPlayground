import numpy as  np
import torch
from torch.utils.data import Dataset, DataLoader
import random

class BinarySequenceDataset(Dataset):
    def __init__(self, num_samples, word_size = 5):
        self.num_samples = num_samples
        self.word_size = word_size
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sequence = [random.randint(0, 1) for _ in range(self.word_size)]  # Generate random 0s and 1s
        sequence = sequence + sequence[::-1]
        sequence_tensor = torch.tensor(sequence, dtype=torch.int)  # Convert to tensor
        sequence_str = ''.join(map(str, sequence))  # Convert to string
        
        return sequence_tensor, sequence_str





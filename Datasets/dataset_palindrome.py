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
class PalindromeDataset(Dataset):
    def __init__(self, seq_len=20):
        self.seq_len = seq_len
        
    def __len__(self):
        return 100000  # effectively infinite

    def __getitem__(self, idx):
        half = np.random.randint(0, 2, size=self.seq_len // 2)
        palindrome = np.concatenate([half, half[::-1]])
        return torch.tensor(palindrome, dtype=torch.long)


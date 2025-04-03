import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader
import random
import math
from torch.nn import TransformerDecoder, TransformerDecoderLayer


class GaussianMixtureDataset(Dataset):
    def __init__(self, num_samples=10000000, means=[0, 5], stds=[1, 1], weights=[0.5, 0.5], seq_length=20, scaling=True):
        """Initialize a dataset that generates samples from a Gaussian mixture model.

        Args:
            num_samples (int): Number of samples to generate.
            means (list): List of means for each Gaussian component.
            stds (list): List of standard deviations for each Gaussian component.
            weights (list): List of mixture weights for each component, must sum to 1.
            seq_length (int): Length of sequences to generate.
            scaling (bool): Whether to apply min-max scaling to the samples.
        """
        self.num_samples = num_samples
        self.means = means
        self.stds = stds
        self.weights = weights
        self.seq_length = seq_length  # Store seq_length for use in __getitem__
        self.scaling = scaling  # Store scaling flag

        # Generate samples from the Gaussian mixture
        self.samples = self._generate_samples()
        self.test_samples = self._generate_samples()
    
    def _generate_samples(self):
        """Generate samples from the Gaussian Mixture Model.

        Returns:
            numpy.ndarray: Array of samples drawn from the mixture distribution.
        """
        components = np.random.choice(len(self.means), size=self.num_samples, p=self.weights)
        samples = np.array([
            np.random.normal(self.means[c], self.stds[c]) for c in components
        ])
        if self.scaling:  # Apply min-max scaling if scaling is true
            min_val = samples.min()
            max_val = samples.max()
            samples = (samples - min_val) / (max_val - min_val)  # Scale to [0, 1]
        return samples
    
    def _discretize_sample(self, sample):
        """Convert a float sample into a list of its digits.

        Args:
            sample (float): A floating point number to discretize.

        Returns:
            list: List of integers representing the digits of the input number.
        """
        sample_str = f"{sample:.19f}"  
        return [int(char) for char in sample_str if char.isdigit()][:self.seq_length]  # Limit to seq_length digits
    
    def __len__(self):
        """Get the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return self.num_samples
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            torch.Tensor: Tensor containing the discretized sample.
        """
        sample = self.samples[idx]
        return torch.tensor(self._discretize_sample(sample), dtype=torch.long)

    def __test_model(self, model):
        """Test a model on the test samples.

        Args:
            model (torch.nn.Module): The model to test.

        Returns:
            torch.Tensor: Model outputs on the test data.
        """
        test_loader = DataLoader(self.test_samples, batch_size=batch_size, shuffle=False)
        model.eval()
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                output = model(data)
                # Process the output as needed
        return output

    def get_vocab_size(self):
        """Get the vocabulary size needed for the autoregressive model.

        Returns:
            int: Size of vocabulary (10 for digits 0-9).
        """
        return 10

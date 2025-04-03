import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader
import random
import math
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import matplotlib.pyplot as plt

class OneDGaussianMixtureDataset(Dataset):
    def __init__(self, seq_length=10, num_components=2, means=None, stds=None, weights=None):
        """
        Args:
            seq_length (int): Number of samples in the dataset.
            num_components (int): Number of Gaussian components in the mixture.
            means (list): List of means for each Gaussian component.
            stds (list): List of standard deviations for each Gaussian component.
            weights (list): Mixing coefficients (should sum to 1).
        """
        self.seq_length = seq_length
        self.num_components = num_components

        # Set default means and stds if not provided
        if means is None:
            self.means = np.linspace(-2, 2, num_components)  # Default means spread across range
        else:
            self.means = np.array(means)

        if stds is None:
            self.stds = np.ones(num_components)  # Default standard deviations of 1
        else:
            self.stds = np.array(stds)

        if weights is None:
            self.weights = np.ones(num_components) / num_components  # Equal probability for each component
        else:
            self.weights = np.array(weights) / np.sum(weights)  # Normalize to sum to 1

    def __len__(self):
        return self.seq_length  # Size of dataset

    def __getitem__(self, idx):
        # Select a Gaussian component based on mixture weights
        component = np.random.choice(self.num_components, p=self.weights)

        # Sample from the selected Gaussian component
        sample = np.random.normal(loc=self.means[component], scale=self.stds[component])

        return torch.tensor(sample, dtype=torch.float32)
    def sample(self, num_samples=1000):
        samples = []
        for _ in range(num_samples):
            component = np.random.choice(self.num_components, p=self.weights)
            sample = np.random.normal(loc=self.means[component], scale=self.stds[component])
            samples.append(sample)
        return np.array(samples)


# Create a dataset instance
dataset = OneDGaussianMixtureDataset(
    seq_length=1000, 
    num_components=3, 
    means=[-2, 0, 2], 
    stds=[0.5, 1.0, 0.3], 
    weights=[0.2, 0.5, 0.3]
)

# Generate samples
samples = dataset.sample(num_samples=5000)

# Plot histogram of the samples
plt.figure(figsize=(8, 5))
plt.hist(samples, bins=50, density=True, alpha=0.6, color='b', edgecolor='black')

# Plot individual Gaussian components
x = np.linspace(min(samples), max(samples), 1000)
for mean, std, weight in zip(dataset.means, dataset.stds, dataset.weights):
    plt.plot(x, weight * (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2),
             linewidth=2, linestyle='dashed', label=f"Component μ={mean}, σ={std}")

plt.title("Gaussian Mixture Model Distribution")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid()
plt.show()


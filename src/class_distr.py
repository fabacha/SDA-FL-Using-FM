import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import seaborn as sns
import numpy as np
from torch.utils.data import Subset, DataLoader
import os
import random
import time
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset, DataLoader


from functools import partial
from matplotlib import pyplot as plt
from collections import defaultdict, deque


idx_to_class = {v: k for k, v in traindata.class_to_idx.items()}

def get_class_distribution(dataloader, idx_to_class):
    count_dict = defaultdict(int)  # Initialize dictionary to count classes

    for _, labels in dataloader:
        for label in labels:
            class_name = idx_to_class[label.item()]
            count_dict[class_name] += 1

    return count_dict

# Function to plot the class distribution
def plot_class_distribution(client_loaders, idx_to_class):
    plt.figure(figsize=(20, 10))
    num_clients = len(client_loaders)

    for i, dataloader in enumerate(client_loaders):
        plt.subplot(2, 5, i + 1)  # Adjust to fit the number of clients
        class_dist = get_class_distribution(dataloader, idx_to_class)
        sns.barplot(x=list(class_dist.keys()), y=list(class_dist.values())).set_title(f'Client {i + 1} Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('Non IID Plot Exp 9')
    plt.show()

# Plot the class distribution for each client
plot_class_distribution(client_loaders, idx_to_class)

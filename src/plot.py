import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import seaborn as sns
import numpy as np
from torch.utils.data import Subset, DataLoader
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset, DataLoader


def plot_client_data_distribution(train_dataset, user_groups_train):
    idx_to_class = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
    
    num_users = len(user_groups_train)
    
    for user_id, indices in user_groups_train.items():
        labels = np.array(train_dataset.targets)[indices.astype(int)]
        label_counts = np.bincount(labels, minlength=10)
        
        plt.figure(figsize=(8, 3))
        sns.barplot(x=list(idx_to_class.values()), y=label_counts)
        plt.title(f"Class Distribution for Client {user_id+1}")
        plt.xlabel("Class")
        plt.ylabel("Number of Samples")
        plt.xticks(rotation=45)
        plt.show()

def plot_client_data_distribution_iid(train_dataset, user_groups_train):
    idx_to_class = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
    
    num_users = len(user_groups_train)
    
    for user_id, indices in user_groups_train.items():
        np.indices = np.array(indices, dtype=np.int32)
        labels = np.array(train_dataset.targets)[np.indices]
        label_counts = np.bincount(labels, minlength=10)
        
        plt.figure(figsize=(8, 3))
        sns.barplot(x=list(idx_to_class.values()), y=label_counts)
        plt.title(f"Class Distribution for Client {user_id+1}")
        plt.xlabel("Class")
        plt.ylabel("Number of Samples")
        plt.xticks(rotation=45)
        plt.show()

# def plot_client_data_distribution(train_dataset, user_groups_train):
#     num_users = len(user_groups_train)
    
#     for user_id, indices in user_groups_train.items():
#         labels = np.array(train_dataset.targets)[indices.astype(int)]
#         label_counts = np.bincount(labels, minlength=10)
        
#         plt.figure(figsize=(10, 5))
#         sns.barplot(x=np.arange(10), y=label_counts)
#         plt.title(f"Class Distribution for Client {user_id}")
#         plt.xlabel("Class")
#         plt.ylabel("Number of Samples")
#         plt.xticks(np.arange(10))
#         plt.show()


# from functools import partial
# from matplotlib import pyplot as plt


# # Assuming traindata is the CIFAR-10 dataset
# idx_to_class = {v: k for k, v in datasets.CIFAR10('./data', download=True, train=True).class_to_idx.items()}

# def get_class_distribution(dataset):
#     count_dict = {k: 0 for k in idx_to_class.values()} # Initialize dictionary

#     for _, label in dataset:
#         for l in label.tolist():  # Convert label tensor to list
#             l = idx_to_class[l]
#             count_dict[l] += 1

#     return count_dict

# def plot_class_distribution(data):
#     plt.figure(figsize=(20, 10))
#     for i, loader in enumerate(data, 1):
#         plt.subplot(2, 5, i)
#         class_dist = get_class_distribution(loader)
#         sns.barplot(x=list(class_dist.keys()), y=list(class_dist.values())).set_title(f'Client {i} Class Distribution')
#         plt.xlabel('Class')
#         plt.ylabel('Count')
#         plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()


# def plot_client_data_distribution(train_dataset, user_groups_train):
#     # Create a DataFrame to store the counts
   

#     num_users = len(user_groups_train)
#     all_data = []

#     for user_id, indices in user_groups_train.items():
#         labels = np.array(train_dataset.targets)[indices.astype(int)]
#         label_counts = np.bincount(labels, minlength=10)
#         for label, count in enumerate(label_counts):
#             all_data.append([user_id, label, count])

#     df = pd.DataFrame(all_data, columns=["User ID", "Class", "Count"])

#     plt.figure(figsize=(20, 10))
#     sns.barplot(x="User ID", y="Count", hue="Class", data=df)
#     plt.title("Class Distribution per Client")
#     plt.xlabel("Client ID")
#     plt.ylabel("Number of Samples")
#     plt.legend(title="Class")
#     plt.show()

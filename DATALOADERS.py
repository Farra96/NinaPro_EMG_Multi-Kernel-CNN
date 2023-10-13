# import scipy
import numpy as np
import os
# import pathlib
import re
# import pandas as pd
# import matplotlib.pyplot as plt
import torch
from torch import nn
# from sklearn import metrics
from torch.utils import data
import torch.nn.functional as F

from math import log2
# from scipy.signal import butter, filtfilt, freqz, iirnotch
from Data_Processing_Utils import windowing_signal, mapping_labels


# %% Personalized dataloader to work with dataframes in order to extract a sample (400 raws) based on
# personalized index & groupby function
class Pandas_Dataset(data.Dataset):

    def __init__(self, df_grouped_by_samples, return_sub=False):
        self.grouped = df_grouped_by_samples
        self.channels = [i for i in df_grouped_by_samples.obj.columns if 'ch' in i]
        self.indices = list(df_grouped_by_samples.indices)
        self.return_sub = return_sub

    def __len__(self):
        return len(self.grouped)

    def __getitem__(self, index):
        picked_smp = self.grouped.get_group(self.indices[index])
        # Picking only the channels columns from a single sample
        sample = torch.tensor(picked_smp.loc[:, self.channels].values).type(torch.float32)
        # picking only one label for each sample
        label = torch.tensor(picked_smp.loc[:, ['label']].head(1).values[0][0]).type(torch.int8)

        if self.return_sub:
            # picking only one subject
            sub = torch.tensor(picked_smp.loc[:, ['sub']].head(1).values[0][0]).type(torch.int8)
            return sample, label, sub
        else:

            # It's missing the part in which I Normalize the data for each subject, or for each subject and channels
            # It's not a good idea to do that on the dataloader while producing results
            return sample, label

# %% dataset for dataframe which retrieve a positive and negative sample
class dataframe_dataset_triplet(data.Dataset):

    def __init__(self, dataframe, groupby_col, target_col, sample_frac=1.0):
        self.df = dataframe
        self.groupby_col = groupby_col
        self.target_col = target_col
        self.sample_frac = sample_frac
        self.channels = [i for i in dataframe.columns if 'ch' in i]

        self.groups = dataframe.groupby(groupby_col)
        self.indices = {i: group.sample(frac=sample_frac).index for i, group in self.groups}
        self.keys = list(self.indices.keys())

        # to delete df.query
        self.labels = np.unique(dataframe['label'])
        self.subjects = np.unique(dataframe[target_col])

        self.sample_indices_by_label_sub = {}
        for label, sub in dataframe[['label', 'sub']].drop_duplicates().values:
            self.sample_indices_by_label_sub[(label, sub)] = np.unique(dataframe.query(f"label == {label} and {'sub'} == {sub}")[
                'sample_index'].values)
        # # TODO: Store less data in dataloaders for triplet and reversal
        #  it should work even with: -> dataframe.query(f"label == {label} and {'sub'} == {sub}")[
        #         'sample_index'].values         ->> so you store less data !!!!!!!!! ADDED NP.UNIQUE()

    def __getitem__(self, idx):
        group_key = self.keys[idx]
        group_indices = self.indices[group_key]
        sample = torch.tensor(self.df.loc[group_indices, self.channels].values).type(torch.float32)
        label = torch.tensor(self.df.loc[group_indices, 'label'].head(1).values[0]).type(torch.int8)
        # sub = self.df.loc[group_indices, self.target_col].head(1).values[0]

        # # take a unique random value from groupby_col (in this case, sample index) after filtered dataframe
        # idx_sample = np.random.choice(self.df.query(f'{self.target_col} != {sub} and label == {label}')[self.groupby_col], size=1)[0]
        # # extract the casual positive sample
        # pos_sample = torch.tensor(self.df.loc[self.df[self.groupby_col] == idx_sample, self.channels].values)\
        #     .type(torch.float32)
        #
        # idx_sample = np.random.choice(self.df.query(f'label != {label}')[self.groupby_col], size=1)[0]
        # neg_sample = torch.tensor(self.df.loc[self.df[self.groupby_col] == idx_sample, self.channels].values)\
        #     .type(torch.float32)

        # After deleting query
        # positive
        # pos_sub = np.random.choice([k for k in self.subjects if k != sub])  # Changed bc in case of one single sub raises error (e.g. test_dataloader)
        pos_sub = np.random.choice(self.subjects)
        pos_indexes = self.sample_indices_by_label_sub[(int(label), pos_sub)]
        pos_sample = torch.tensor(self.groups.get_group(np.random.choice(pos_indexes, size=1)[0])[self.channels].values) \
            .type(torch.float32)
        # negative
        neg_lbl = np.random.choice([k for k in self.labels if k != label])
        neg_indexes = self.sample_indices_by_label_sub[(neg_lbl, np.random.choice(self.subjects))]
        neg_sample = torch.tensor(
                self.groups.get_group(np.random.choice(neg_indexes, size=1)[0])[self.channels].values) \
                .type(torch.float32)

        return sample, label, pos_sample, neg_sample

    def __len__(self):
        return len(self.keys)

#%%
class dataframe_dataset_JS(data.Dataset):

    def __init__(self, dataframe, groupby_col, target_col, sample_frac=1.0):
        self.df = dataframe
        self.groupby_col = groupby_col
        self.target_col = target_col
        self.sample_frac = sample_frac
        self.channels = [i for i in dataframe.columns if 'ch' in i]

        self.groups = dataframe.groupby(groupby_col)
        self.indices = {i: group.sample(frac=sample_frac).index for i, group in self.groups}
        self.keys = list(self.indices.keys())

        # to delete df.query
        self.labels = np.unique(dataframe['label'])
        self.subjects = np.unique(dataframe[target_col])

        self.sample_indices_by_label_sub = {}
        for label, sub in dataframe[['label', 'sub']].drop_duplicates().values:
            self.sample_indices_by_label_sub[(label, sub)] = dataframe.query(f"label == {label} and {'sub'} == {sub}")[
                'sample_index'].values

    def __getitem__(self, idx):
        group_key = self.keys[idx]
        group_indices = self.indices[group_key]
        sample = torch.tensor(self.df.loc[group_indices, self.channels].values).type(torch.float32)
        label = torch.tensor(self.df.loc[group_indices, 'label'].head(1).values[0]).type(torch.int8)
        sub = self.df.loc[group_indices, self.target_col].head(1).values[0]

        # # take a unique random value from groupby_col (in this case, sample index) after filtered dataframe
        # idx_sample = np.random.choice(self.df.query(f'{self.target_col} != {sub} and label == {label}')[self.groupby_col], size=1)[0]
        # # extract the casual positive sample
        # pos_sample = torch.tensor(self.df.loc[self.df[self.groupby_col] == idx_sample, self.channels].values)\
        #     .type(torch.float32)
        #
        # idx_sample = np.random.choice(self.df.query(f'label != {label}')[self.groupby_col], size=1)[0]
        # neg_sample = torch.tensor(self.df.loc[self.df[self.groupby_col] == idx_sample, self.channels].values)\
        #     .type(torch.float32)

        # After deleting query
        # positive
        # pos_sub = np.random.choice([k for k in self.subjects if k != sub])  # Changed bc in case of one single sub raises error (e.g. test_dataloader)
        pos_sub = np.random.choice(self.subjects)
        pos_indexes = self.sample_indices_by_label_sub[(int(label), pos_sub)]
        pos_sample = torch.tensor(self.groups.get_group(np.random.choice(pos_indexes, size=1)[0])[self.channels].values) \
            .type(torch.float32)


        return sample, label, pos_sample

    def __len__(self):
        return len(self.keys)

# %% General grouped dataset for dataloader
class GroupbyDataset(data.Dataset):
    def __init__(self, dataframe, groupby_col, target_col, sample_frac=1.0):
        self.dataframe = dataframe
        self.groupby_col = groupby_col
        self.target_col = target_col
        self.sample_frac = sample_frac
        self.channels = [i for i in dataframe.columns if 'ch' in i]

        self.groups = dataframe.groupby(groupby_col)
        self.indices = list(self.groups.indices)
        # self.indices = {i: group.sample(frac=sample_frac).index for i, group in self.groups}
        # self.keys = list(self.indices.keys())

    def __getitem__(self, idx):
        # group_key = self.keys[idx]
        # group_indices = self.indices[group_key]
        pick_sample = self.groups.get_group(self.indices[idx])
        sample = torch.tensor(pick_sample.loc[:, self.channels].values).type(torch.float32)
        label = torch.tensor(pick_sample.loc[:, 'label'].head(1).values[0]).type(torch.int8)
        sub = pick_sample['sub'].head(1).values[0]

        return sample, label

    def __len__(self):
        return len(self.indices)


# %% KL and JS divergence
class JSDLoss(torch.nn.Module):
    """
    Jensen-Shannon Divergence loss function for matching the
    distributions of embeddings produced by two networks.
    """

    def __init__(self):
        super(JSDLoss, self).__init__()

    def forward(self, e1, e2):
        """
        Computes the Jensen-Shannon divergence between the distributions
        of embeddings e1 and e2.

        Args:
        - e1: Tensor of shape (batch_size, embedding_size)
        - e2: Tensor of shape (batch_size, embedding_size)

        Returns:
        - loss: Scalar tensor representing the JSD between e1 and e2
        """

        # Concatenate the embeddings along the batch dimension
        e = torch.cat((e1, e2), dim=0)

        # Compute the logits (not normalized log probabilities) for each embedding
        logits = F.log_softmax(e, dim=0)

        # Compute the normalized probabilities for each embedding
        p = torch.exp(logits)
        p1, p2 = torch.split(p, split_size_or_sections=e1.size(0), dim=0)

        # Compute the mean probability distributions for e1 and e2
        m = 0.5 * (p1 + p2)

        # Compute the JSD between e1 and e2
        loss = 0.5 * (F.kl_div(p1, m, reduction='batchmean') + F.kl_div(p2, m, reduction='batchmean'))

        return loss

class CenterLoss(torch.nn.Module):
    def __init__(self, num_classes, feat_dim, alpha=0.5, device = None):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = alpha

        # Create center parameter for each class
        self.centers = torch.nn.Parameter(torch.randn(num_classes, feat_dim))
        self.device = device

    def forward(self, x, labels):
        """
        Calculate center loss
        Args:
            x: Feature embeddings (batch_size, feat_dim)
            labels: Ground truth labels (batch_size,)

        Returns:
            Center loss value
        """

        batch_size = x.size(0)


        # Gather the centers corresponding to the labels
        centers_batch = self.centers[labels].to(self.device)


        # Compute the center loss
        center_loss = torch.sum((x - centers_batch) ** 2) / (2.0 * batch_size)

        return center_loss

    def update_centers(self, x, labels):
        """
        Update the centers using an exponential moving average (EMA).
        Args:
            x: Feature embeddings (batch_size, feat_dim)
            labels: Ground truth labels (batch_size,)
        """
        # Calculate the count of samples for each class
        count = torch.bincount(labels, minlength=self.num_classes).float()

        # Calculate the mean embeddings for each class
        sum_embeddings = torch.zeros_like(self.centers)
        for i in range(self.num_classes):
            mask = labels == i
            if mask.any():
                sum_embeddings[i] = torch.mean(x[mask], dim=0)

        # Update the centers using EMA
        self.centers = torch.nn.Parameter(self.alpha * sum_embeddings + (1 - self.alpha) * self.centers)


# %% OLD PART

def calculate_jensen_shannon_divergence(model1, model2):
    # Get the weights of the second nn.Linear layer for both models
    weights1 = model1.model[-2].weight.view(-1)
    weights2 = model2.model[-2].weight.view(-1)

    # Normalize the weights
    weights1 = nn.Softmax(weights1)
    weights2 = nn.Softmax(weights2)

    # Calculate Jensen-Shannon Divergence
    m = 0.5 * (weights1 + weights2)
    jsd = 0.5 * (F.kl_div(weights1, m) + F.kl_div(weights2, m))

    return jsd
# def calculate_jensen_shannon_divergence(model1, model2):
#     # Get the weights of both models
#     weights1 = torch.cat([param.view(-1) for param in model1.parameters()])
#     weights2 = torch.cat([param.view(-1) for param in model2.parameters()])
#
#     # Normalize the weights
#     weights1 = F.softmax(weights1, dim=0)
#     weights2 = F.softmax(weights2, dim=0)
#
#     # Calculate Jensen-Shannon Divergence
#     m = 0.5 * (weights1 + weights2)
#     jsd = 0.5 * (F.kl_div(weights1, m) + F.kl_div(weights2, m))
#
#     return jsd

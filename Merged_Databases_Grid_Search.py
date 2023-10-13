import csv
import random
import sys
import os

from sklearn import metrics

os.chdir('/home/ricardo/')

import numpy as np  # numpy library
import torch
import torch.nn as nn
import pandas as pd
import argparse

from scipy.stats import loguniform

# CUSTOM LYBRARIES
from DATALOADERS import Pandas_Dataset
from Training_type import random_GS
from Data_Processing_Utils import Norm_each_sub_by_own_param, train_val_test_split_df

#%% Delete FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # -> if pandas is bothering you

# %% Loading and merging datasets
# Choose which window to use (so which dataframe to load)
wnd_len = 300

# DB2            -> Left handed already deleted at creation
df_DB2 = pd.read_csv(f'/home/ricardo/DB2/Dataframe/dataframe_wnd_{wnd_len}_filt_20_500.csv',
                     dtype={'sub': np.int8, 'label': np.int8, 'rep': np.int8, 'sample_index': np.uint32})

# DB7
df_DB7 = pd.read_csv(f'/home/ricardo/DB7/Dataframe/dataframe_wnd_{wnd_len}_filt_20_500.csv',
                     dtype={'sub': np.int8, 'label': np.int8, 'rep': np.int8, 'sample_index': np.uint32})
############ DROPPING LEFT HANDED ######################
left_hand_DB7 = [1, 4, 5, 14]
df_DB7.drop(df_DB7[df_DB7['sub'].isin(left_hand_DB7)].index, inplace=True)
###########################################################################

# DB3 (# for some reason subject 1 has done less exercise) -> [ 0,  4,  7,  8,  9, 10, 11, 12, 13]
df_DB3 = pd.read_csv(f'/home/ricardo/DB3/Dataframe/dataframe_wnd_{wnd_len}_filt_20_500.csv',
                     dtype={'sub': np.int8, 'label': np.int8, 'rep': np.int8, 'sample_index': np.uint32})

############ DROPPING LEFT HANDED SUBJECT
left_hand_DB3 = [2, 5, 6]
df_DB3.drop(df_DB3[df_DB3['sub'].isin(left_hand_DB3)].index, inplace=True)
##############################################################################################
# Setting ch9 and ch10 of subjects 6 and 7 to zero (not used and set as -1 for space problems)
# df_DB3.loc[df_DB3['sub'] == 6, 'ch9'] = 0     # -> commented because I already dropped it as left-handed
# df_DB3.loc[df_DB3['sub'] == 6, 'ch10'] = 0
df_DB3.loc[df_DB3['sub'] == 7, 'ch9'] = 0
df_DB3.loc[df_DB3['sub'] == 7, 'ch10'] = 0

# %% Merging dataset and reindexing samples and subjects
# Changing sub in the order -> DB2 -> DB7 -> DB3
df_DB7['sub'] = df_DB7['sub'] + np.max(df_DB2['sub'])
df_DB3['sub'] = df_DB3['sub'] + np.max(df_DB7['sub'])

# Concatenate them
df = pd.concat([df_DB2, df_DB7, df_DB3])
if len(df_DB2) + len(df_DB3) + len(df_DB7) != len(df):
    print('Something went wrong during concatenation!')
# Free space
del df_DB2, df_DB7, df_DB3

# Fixing the "sample_index" column that will be picked from dataloaders
# calculate the maximum value for the new column
n_samples = len(df) // wnd_len

# create an array of values that are repeated WND_LEN times
samples_index = np.repeat(range(n_samples), wnd_len)

# Reindexing samples
df['sample_index'] = samples_index

# Save dataframe
df.to_csv(path_or_buf=f'/home/ricardo/DB2+DB7+DB3/Dataframe/dataframe_wnd_{wnd_len}.csv')

#%% Script to be run from console for Grid_search
if __name__ == '__main__':
    #Delete FutureWarning
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)  # -> if pandas is bothering you

    # %% Get params from prompt
    parser = argparse.ArgumentParser(description='Configurations to train models.')

    parser.add_argument('-type', '--NORM_TYPE', help='Normalization_Type', type=str, default='z_score')
    parser.add_argument('-mode', '--NORM_MODE', help='Normalization_Mode', type=str, default='channel')
    parser.add_argument('-rec', '--RECT', help='Absolute value or normal', default='True', type=str)
    parser.add_argument('-wnd', '--WND_LEN', help='Windows length', type=int, default=300)

    args = parser.parse_args()
    NORM_TYPE = str(args.NORM_TYPE)
    NORM_MODE = str(args.NORM_MODE)
    RECT = args.RECT.lower() == 'true'      # Became false if you pass -rec 'False'
    WND_LEN = args.WND_LEN

    print(f'WND_LEN  :  --> {WND_LEN}\n'
          f'NORM_MODE:  --> {NORM_MODE}\n'
          f'NORM_TYPE:  --> {NORM_TYPE}\n'
          f'RECTIFY  :  --> {RECT}\n')

    # %% Load dataframe
    # Choose which window to use (so which dataframe to load)
    database = '/home/ricardo/DB2+DB7+DB3/'
    df = pd.read_csv(f'{database}/Dataframe/dataframe_wnd_{WND_LEN}.csv',
                     dtype={'sub': np.int8, 'label': np.int8, 'rep': np.int8, 'sample_index': np.uint32})
    # Making it lighter
    n_channels = [cx for cx in df.columns if 'ch' in cx]
    df.loc[:, n_channels] = df.loc[:, n_channels].astype(np.float32)

    # Classes selected
    exe_labels = ['Medium wrap', 'Lateral', 'Extensions type', 'Tripod', 'Power sphere', 'Power disk', 'Prismatic pitch',
                  'Index Extension', 'Thumb Adduction', 'Prismatic 4 fingers', 'Wave in', 'Wave out', 'Fist', 'Open hand']

    # %% Normalization
    df = Norm_each_sub_by_own_param(df, norm_type=NORM_TYPE, mode=NORM_MODE, rectify=RECT)
    # INFOS: SUBJECT 69 CHANNEL 9 AND 10 ARE ALL ZEROS (sub 69 is the sub number 7 of  DB3, amputee and no space for ch)
    # so if u try to do z_score, you will have nan in the channel of that subject
    df = df.fillna(0)

    # %% Split dataframe
    # Splitting Train, val and teste based on Rep
    split_mode = 'rep'
    train_val_test_rep = [[2, 4, 6], [1, 5], [3]]  # [[1, 2, 5, 6], [4], [3]]
    df_train, df_val, df_test = train_val_test_split_df(df, mode=split_mode, manual_sel=train_val_test_rep)

    # Splitting Train, val and teste based on Sub
    # split_mode = 'sub'    # Selected subject 5->175[cm]x75[kg]
    # sub = 5
    # val_list = np.random.choice([x for x in n_subject.tolist() if x != sub], size=10).tolist()
    # tr_list = [i for i in n_subject if i not in val_list and i != sub]
    #
    #
    # df_train, df_val, df_test = train_val_test_split_df(df, mode=split_mode, manual_sel=[tr_list, val_list, [sub]])

    valid_group = np.unique(df_val[split_mode])
    test_group = np.unique(df_test[split_mode])[
        0]  # Usually only one rep or sub is used for testing, modify it accordingly

    if len(df_train) + len(df_val) + len(df_test) != len(df):
        IndexError(
            'Something went wrong when splitting dataframe! Some data are not part of either the train, val and test')
    del df
    # %% Dataset from which creates dataloaders
    train_set = Pandas_Dataset(df_train.groupby('sample_index'))
    valid_set = Pandas_Dataset(df_val.groupby('sample_index'))
    test_set = Pandas_Dataset(df_test.groupby('sample_index'))


    # %% Grid search initialization
    device = torch.device('cuda')

    # Dimensions of parallel kernels for MKCNN
    kernels_gap = [g for g in range(0, 3*round(WND_LEN/20),  round(WND_LEN/20))]
    kernel_sizes = np.full((3, 5, 2), [1, 3])
    for j in range(3):
        for i in range(5):
            kernel_sizes[j][i][0] = (10*(i + 1) + kernels_gap[j])
        n_classes = 14

    grid = {'net': {'N_multik': [16, 32, 64], 'N_Conv_conc': [64, 128, 256], 'N_SepConv': [64, 128, 256],
                    'Kernel_multi_dim': [kernel_sizes[0], kernel_sizes[1], kernel_sizes[2]],
                    'Kernel_Conv_conc': [1, 3],
                    'act_func': [nn.ReLU(), nn.LeakyReLU(), nn.ELU(), nn.PReLU(), nn.GELU],
                    'Pool_Type': [nn.MaxPool2d, nn.AvgPool2d], 'wnd_len': [WND_LEN]
                    },
            'learning': {'n_models': [50],
                         'num_epochs': [100],
                         'lr': loguniform.rvs(1e-4, 1e-2, size=10),
                         'batch_size': [128, 256, 512],
                         # 'folds': [5],    # In case of cross_validation, but here I use a manual validation set
                         'opt': ['Adam'],
                         'loss_fn': [nn.CrossEntropyLoss(reduction='mean').to(device)],
                         'device': [device],
                         'num_workers': [4]
                         }
            }


# %% Grid_search

# Grid_search, divided by repetition (total 6): 4 for training one for validation one for test
random_GS(grid=grid, train_set=train_set, valid_set=valid_set, test_set=test_set, cm_labels=exe_labels,
          path_to_save=f'/home/ricardo/DB2+DB7+DB3/grid_search_{WND_LEN}_{NORM_TYPE}_{RECT}')

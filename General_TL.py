import argparse
import csv
import os
import sys
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import loguniform
from sklearn import metrics
from torch.utils import data



# Custom Libraries
sys.path.append('/home/ricardo/MKCNN')

from Data_Processing_Utils import Norm_each_sub_by_own_param, train_val_test_split_df, PlotLoss, plot_confusion_matrix

from DATALOADERS import dataframe_dataset_triplet, Pandas_Dataset, dataframe_dataset_JS

from Training_type import  train_model_standard, train_model_reversal_gradient, train_model_JS, \
    train_model_triplet, pre_train_model_triplet

from MODELS import MKCNN, MKCNN_grid, MKCNN_grid_AN, \
    MultiKernelConv2D_grid, TCN_ch_sp_ATT, get_TL_results


# INFOS DB3:
# -left hand: [2, 5, 6]
# - SUB 4: -> is over the double from second maximum (worse) DASH SCORE, consider drop him.
# - 66 Missing files after formatting (66 missing 'SubXeYrepZ')
# - SUB: 1 has done fewer exercise: 9 / 14 (may have wrong labels?)
# - SUB7: channel 9 and 10 has same min and max, not used -> 0% forearm

# %% Delete FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # -> if pandas is bothering you

# %% Get params from prompt
parser = argparse.ArgumentParser(description='Configurations to train models.')
# parser.add_argument('-sub', '--N_SUB', help='Subject_to_test',type=int, default=0)
parser.add_argument('-mode', '--NORM_MODE', help='Norm_Mode',type=str, default='channel')
parser.add_argument('-norm', '--NORM_TYPE', help='Normalization_Type', type=str, default='z_score')
parser.add_argument('-rec', '--RECT', help='Absolute value or normal', default='True', type=str)
parser.add_argument('-wnd', '--WND_LEN', help='Windows length', type=int, default=300)
parser.add_argument('-db', '--DATABASE', help='Database to pick', type=str, default='DB2+DB3+DB7')
parser.add_argument('-type', '--TRAIN_TYPE', help = 'Train typology', type=str, default='Standard')

parser.add_argument('-freeze', '--FREEZE', help='Layers to freeze', type=str, default='last')

args = parser.parse_args()

# N_SUB = int(args.N_SUB)
TRAIN_TYPE = str(args.TRAIN_TYPE)
DATABASE = str(args.DATABASE)
WND_LEN = int(args.WND_LEN)
NORM_MODE = str(args.NORM_MODE)
NORM_TYPE = str(args.NORM_TYPE)
RECT = args.RECT.lower() == 'true'      #    Became false if you pass -rec 'False'

FREEZE = args.FREEZE

#%% Manual selection
# TRAIN_TYPE = 'Reversal'
# DATABASE = 'DB2+DB7+DB3'
# WND_LEN = 300
# NORM_MODE = 'channel'
# NORM_TYPE = 'z_score'
# RECT = True
# ACT_FUN = 'relu'
# FREEZE = 'middle'

print(f'TRAIN TYPE: --> {TRAIN_TYPE}\n'
      f'DATABASE :  --> {DATABASE}  \n'
      f'WND_LEN  :  --> {WND_LEN}   \n'
      f'NORM_MODE:  --> {NORM_MODE} \n'
      f'NORM_TYPE:  --> {NORM_TYPE} \n'
      f'RECTIFY  :  --> {RECT}      \n'
      f'FREEZE   :  --> {FREEZE}'   )

# %% Database path
# DATABASE = 'DB3'
# WND_LEN = 300
database = f'/home/ricardo/{DATABASE}/'
exe_labels = ['Medium wrap', 'Lateral', 'Extensions type', 'Tripod', 'Power sphere', 'Power disk', 'Prismatic pitch',
              'Index Extension', 'Thumb Adduction', 'Prismatic 4 fingers', 'Wave in', 'Wave out', 'Fist', 'Open hand']

device = torch.device('cuda')

# %% Left-Handed & Corrections
# Left-handed dropped during creation of dataframe
df = pd.read_csv(database + f'Dataframe/dataframe_wnd_{WND_LEN}.csv',
                 dtype={'sub': np.int8, 'label': np.int8, 'rep': np.int8, 'sample_index': np.uint32})
# Making it lighter
n_channels = [cx for cx in df.columns if 'ch' in cx]
df.loc[:, n_channels] = df.loc[:, n_channels].astype(np.float32)

# %% Normalization
# NORM_TYPE = 'z_score'
# RECT = True
# NORM_MODE = 'channel'

filename = f'wnd_{WND_LEN}_{NORM_TYPE}_{NORM_MODE}_rect_{RECT}'
if TRAIN_TYPE == 'Reversal':
    filename = filename +'_5_Groups'

df = Norm_each_sub_by_own_param(df, norm_type=NORM_TYPE, mode=NORM_MODE, rectify=RECT)

# INFOS & CORRECTIONS:
# - SUBJECT 69 CHANNEL 9 AND 10 ARE ALL ZEROS (sub 69 is the sub number 7 of  DB3, amputee and no space for them (0% of forearm).
# Doing Z_score for each subject and channel separately, ends up with nan in the ch9 and ch10 of this subject.
# - # In DB3 (for some reason subject 1 has done less exercise) -> [ 0,  4,  7,  8,  9, 10, 11, 12, 13]
# It doesn't seem a wrong labeling, because the numbers for each gesture are within the normal range, but I decided to drop him.
# He corresponds to subject 63 in the merged DB2+DB7+DB3

df = df.fillna(0)
if DATABASE == 'DB3':
    df = df.query('sub != 1')
    left_hand_DB3 = [2, 5, 6]
    df.drop(df[df['sub'].isin(left_hand_DB3)].index, inplace=True)      # Drop left-handed
elif DATABASE == 'DB2+DB7+DB3':
    df = df.query('sub != 63')

#TODO: insert him (and maybe subject 69 aka subject 7 of DB3 with 0 forearm in sub_to_test


n_subject, n_rep = np.unique(df['sub']), np.unique(df['rep'])

# Select sub to test
random.seed(29)
sub_to_test = random.sample(list(n_subject), k= round(len(n_subject)/5))  # for TL framework
# [42, 6, 28, 47, 65, 70, 21, 7, 38, 71, 31]    -> 11 subject in merge
# [9]    ->  DB3
sub_to_test_df = df[df['sub'].isin(sub_to_test)]

# This is added because, in function of the DATABASE used, there are different number of subjects which involves different
# classes for the domain classifier. To build the model properly before to load-weights, you need to know their number.
if TRAIN_TYPE == 'Reversal':
    # Drop sub used for test
    df = df[~(df['sub'].isin(sub_to_test))]
    # Splitting Rep
    split_mode = 'rep'
    train_val_test_rep = [[2, 4, 6], [1, 5], [3]]  # [[1, 2, 5, 6], [4], [3]]
    df_train, _, _, = train_val_test_split_df(df, mode=split_mode, manual_sel=train_val_test_rep)
    num_dom = len(np.unique(df_train['sub']))   # Number of classes for domain classifier
    del df_train


del df, _

#%% Grid
kernels_gap = [g for g in range(0, 3 * round(WND_LEN / 20), round(WND_LEN / 20))]
kernel_sizes = np.full((3, 5, 2), [1, 3])
for j in range(3):
    for i in range(5):
        kernel_sizes[j][i][0] = (round(WND_LEN / 20) * (i + 1) + kernels_gap[j])
n_classes = 14

net= {'N_multik': 32, 'N_Conv_conc': 128, 'N_SepConv': 256,
                'Kernel_multi_dim': kernel_sizes[0],
                'Kernel_Conv_conc': 1,
                'act_func': nn.ReLU(),
                'Pool_Type': nn.MaxPool2d,
                'wnd_len': WND_LEN,
                 # The following key is used only for Adversarial Network, to decide where the domain classifier starts
                'Output2': nn.Flatten  # -> It will start at the first linear layer
                # 'Output2': MultiKernelConv2D_grid  # -> It will start after the multikernel stage
                }

# %% Grid initialization
if TRAIN_TYPE == 'Standard' or TRAIN_TYPE == 'Triplet' or TRAIN_TYPE == 'JS':
    model = MKCNN_grid(net)

elif TRAIN_TYPE == 'Reversal':
    if '5_Groups' in filename:
        model = MKCNN_grid_AN(net, num_domains=5)        # Previously split subjects in 5 groups during training
    else:
        model = MKCNN_grid_AN(net, num_domains=num_dom)  # depending on dataframe number of subjects
    # The following change in "filename" variable is added because the second-head of Adversarial Network can start from
    # "middle" or "last". Basically, if you made the domain classifier start from "middle", the pre-trained model
    # has learnt only the part 1 of the model to be invariant with respect to domain, so you want to re-train
    # the model from "middle". Same concepts for the "last": you don't want to re-train the model from "middle" if,
    # in part 2 nd part 3 of the article you have learnt invariant-feature. Indeed, the selection of freezing method for
    # Transfer Learning automatically select the correspondent model's structure to load weights from.
    filename = filename + f'_{FREEZE}'

elif TRAIN_TYPE == 'TCN_SAM_CAM':
    l2_factor = 0.2
    model = TCN_ch_sp_ATT(wnd_len=WND_LEN, n_classes=n_classes, l2_lambda=l2_factor)


# %% TF results
# Load weights
model.load_state_dict(torch.load(f'{database}/{TRAIN_TYPE}/Cross_rep/Best_States/state_dict_{filename}.pth'))
model = model.to(device)

total_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
print(f'Model loaded with Num params: {total_params}')

# %% Freezing params:
# Freezing parameters
if TRAIN_TYPE != 'TCN_SAM_CAM':
    if FREEZE == 'last':
        model.freeze_multik()           # Freeze first part
        batch_size = 64
        for layer in model.model:       # Leaving unfrozen only last 2 Linear Layers
            if not isinstance(layer, nn.Linear):
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                break

    elif FREEZE == 'middle':
        batch_size = 128
        model.freeze_multik()            # Freeze first part only

# Only for TCN_SAM_CAM because it is different
else:
    batch_size = 64
    # Freeze all
    for param in model.parameters():        # Freeze all
        param.requires_grad = False
    # Unfreeze last
    for layer in model.conv_block:          # Unfreeze last linear layers
        if isinstance(layer, nn.Linear):
            for param in layer.parameters():
                param.requires_grad = True



num_param_tl = sum([p.numel() for p in model.parameters() if p.requires_grad])
print(f'Freezing {total_params - num_param_tl} parameters -> TL model has now {num_param_tl} trainable parameters')

# %% Param for TL dataloader
num_workers = 8
params = {'batch_size': batch_size,
          'shuffle': True,
          # 'sampler': sampler,
          'num_workers': num_workers,
          'drop_last': False}


# Param for TL training
optimizer_tl = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-4)
cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean').to(device)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_tl, factor=.2,
                                                       patience=5, verbose=True, eps=1e-6)
#%% Cycle over sub_to_test and over reps
# -> Here it is possible to change the training type for fitting new user.
# For example, you can apply Adversarial Network with reversal gradient to learn Repetition-invariant-feature for the new subject.
cc = 1
n_rep = [2, 1, 4, 5, 6, 3]
for subx in sub_to_test:
    print(70 * '#', 5 * '\n', f'Testing sub Number: -> {cc} / {len(sub_to_test)}', 5 * '\n', 70 * '#')
    df = sub_to_test_df[sub_to_test_df['sub'] == subx]
    cc = cc + 1
    for r in range(0, len(n_rep)):
        reps = n_rep[:r]
        get_TL_results(model=model, tested_sub_df=df, device=device,
                       list_of_rep_to_use=reps, exe_labels=exe_labels,
                       num_epochs=(10 * len(reps)), loss_fn=cross_entropy_loss, optimizer=optimizer_tl, **params,
                       scheduler=scheduler, filename=filename,
                       path_to_save=f'{database}/{TRAIN_TYPE}/TL/{FREEZE}/')

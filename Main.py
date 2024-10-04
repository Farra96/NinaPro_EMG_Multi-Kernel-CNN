import os
import pandas as pd
import torch
import torch.nn as nn
import sys
import numpy as np
import random
import argparse

import time
import csv
import copy

from scipy.stats import loguniform
from sklearn import metrics
from torch.utils import data

# Custom Libraries
if os.getcwd() != os.path.abspath('/home/riccardo/MKCNN'):
    os.chdir('/home/riccardo/MKCNN')
sys.path.append('/home/ricardo/MKCNN')

from Data_Processing_Utils import Norm_each_sub_by_own_param, Norm_each_sub_label_by_own_param, \
    train_val_test_split_df, PlotLoss, plot_confusion_matrix

from DATALOADERS import dataframe_dataset_triplet, Pandas_Dataset, dataframe_dataset_JS

from Training_type import  train_model_standard, train_model_reversal_gradient, train_model_JS, \
    train_model_triplet, pre_train_model_triplet

from MODELS_2 import MKCNN, MKCNN_grid, MKCNN_grid_AN, MKCNN_grid_ATTN, \
    MultiKernelConv2D_grid, TCN_ch_sp_ATT, get_TL_results

# TODO: 
# AdaBatch normalitation:       -> Probably not necessary because the way data are normalize before to be fed to the network
# - Calculayte mean and variance from each subject in the training
# - During inference, load BN statistic of subjects in the batch
# - During TL, calculate new subject BN statistics before re-train
# Normalization technique +lowpass  -> https://ieeexplore.ieee.org/abstract/document/10285513 -> introduced to be tested (doesn't seems to work at all)
# Build a subject-specific model to check performance 

# INFOS DB3:
# -left hand: [2, 5, 6]
# - SUB 4: -> is over the double from second maximum (worse) DASH SCORE, consider take him out
# - 66 Missing files after formatting (66 missing 'SubXeYrepZ')  -> SUB: 1 has done fewer exercise: 9 / 14 
# - SUB7: channel 9 and 10 has same min and max, not used -> 0% forearm -> fill with 0 as they are saved as -1

# %% Delete FutureWarning
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)  # -> if pandas is bothering you

# %% Get params from prompt
parser = argparse.ArgumentParser(description='Configurations to train models.')
# parser.add_argument('-sub', '--N_SUB', help='Subject_to_test',type=int, default=0)
parser.add_argument('-db', '--DATABASE', help='Database to pick', type=str, default='DB2+DB7+DB3')
parser.add_argument('-wnd', '--WND_LEN', help='Windows length', type=int, default=300)
parser.add_argument('-rec', '--RECT', help='Absolute value or normal', default='True', type=str)
parser.add_argument('-mode', '--NORM_MODE', help='Norm_Mode',type=str, default='channel')
parser.add_argument('-norm', '--NORM_TYPE', help='Normalization_Type', type=str, default='z_score')
parser.add_argument('-split', '--SPLIT', help='How to split dataframe',type=str, default='rep')

parser.add_argument('-type', '--TRAIN_TYPE', help = 'Train typology', type=str, default='Standard')
parser.add_argument('-epochs', '--EPOCHS', help='Number of Epochs', type=int, default=100)
parser.add_argument('-act', '--ACT_FUN', help = 'Activation_Function', type=str, default='lrelu')
parser.add_argument('-freeze', '--FREEZE', help='Layers to freeze', type=str, default='last')
parser.add_argument('-gamma', '--GAMMA', help='Weight of second loss function', type=float, default=0.5)
parser.add_argument('-groups', '--GROUPS', help='Divide or not the subjects in 5 groups for reversal', action='store_true')
parser.add_argument('-amputee', '--AMPUTEE', help='Exploit Reversal between healty and amputee subs', action='store_true')
parser.add_argument('-dom_val_loss', '--DOM_VAL_LOSS', help='Exploit the DOMAIN loss on validation for model to be selected', action='store_true')

parser.add_argument('-dev', '--DEVICE', help='Device to use (cuda or cpu)', type=int, default=0)

args = parser.parse_args()

# N_SUB = int(args.N_SUB)
DATABASE = str(args.DATABASE)
WND_LEN = int(args.WND_LEN)
RECT = args.RECT.lower() == 'true'      #  Became false if you pass -rec 'False'
NORM_MODE = str(args.NORM_MODE)
NORM_TYPE = str(args.NORM_TYPE)
SPLIT = str(args.SPLIT)

TRAIN_TYPE = str(args.TRAIN_TYPE)
EPOCHS = int(args.EPOCHS)
ACT_FUN = str(args.ACT_FUN)
FREEZE = str(args.FREEZE)
GAMMA = float(args.GAMMA)
GROUPS = True if args.GROUPS else False
AMPUTEE = True if args.AMPUTEE else False
DOM_VAL_LOSS = True if args.DOM_VAL_LOSS else False

device = torch.device(f'cuda:{args.DEVICE}' if args.DEVICE >= 0 else 'cpu')
#%% Manual selection
# DATABASE = 'DB2+DB7+DB3'
# WND_LEN = 300
# NORM_MODE = 'channel'
# NORM_TYPE = 'z_score'
# RECT = True
# SPLIT = 'rep'

# TRAIN_TYPE = 'Reversal'
# EPOCHS = 5
# ACT_FUN = 'lrelu'
# FREEZE = 'last'
# GAMMA = 0.9
# GROUPS = False
# AMPUTEE = False
# DOM_VAL_LOSS = False
# device = torch.device('cuda:0')

print('\n\nSETTINGS:                    \n'
      f'DATABASE        :  --> {DATABASE}  \n'
      f'WND_LEN         :  --> {WND_LEN}   \n'
      f'RECTIFY         :  --> {RECT}      \n'
      f'NORM_MODE       :  --> {NORM_MODE} \n'
      f'NORM_TYPE       :  --> {NORM_TYPE} \n'
      f'SPLIT           :  --> {SPLIT}     \n'
      f'EPOCHS          :  --> {EPOCHS}    \n'
      f'TRAIN TYPE      : --> {TRAIN_TYPE}\n'
      f'ACT_FUN         :  --> {ACT_FUN}   \n'
      f'FREEZE          :  --> {FREEZE}    \n'
      f'GAMMA           :  --> {GAMMA}     \n'
      f'GROUPS          :  --> {GROUPS}    \n'
      f'AMPUTEE         :  --> {AMPUTEE}   \n'
      f'DOM VAL LOSS    :  --> {DOM_VAL_LOSS}\n'
      f'DEVICE          :  --> {device}')   

# %% Database path
database = f'/home/riccardo/EMG_data/{DATABASE}/'
exe_labels = ['Medium wrap', 'Lateral', 'Extensions type', 'Tripod', 'Power sphere', 'Power disk', 'Prismatic pitch',
              'Index Extension', 'Thumb Adduction', 'Prismatic 4 fingers', 'Wave in', 'Wave out', 'Fist', 'Open hand']

# torch.cuda.set_device(1)
# device = torch.device('cuda:0')
# device = torch.device("cpu")

filename = f'wnd_{WND_LEN}_{NORM_TYPE}_{NORM_MODE}_rect_{RECT}'

# %% Left-Handed & Corrections
# Left-handed dropped during creation of dataframe
df = pd.read_csv(database + f'Dataframe/dataframe_wnd_{WND_LEN}.csv',
                 dtype={'sub': np.int8, 'label': np.int8, 'rep': np.int8, 'sample_index': np.uint32})
# Making it lighter
n_channels = [cx for cx in df.columns if 'ch' in cx]
df.loc[:, n_channels] = df.loc[:, n_channels].astype(np.float32)

# %% Normalization
# df = Norm_each_sub_by_own_param(df, norm_type=NORM_TYPE, mode=NORM_MODE, rectify=RECT)
df = Norm_each_sub_label_by_own_param(df, mode='sub', norm_type=NORM_TYPE, rectify=RECT)

# INFOS & CORRECTIONS:
# 1) -> SUBJECT 69 CHANNEL 9 AND 10 ARE ALL ZEROS (sub 69 is the sub number 7 of  DB3, amputee and no space for them (0% of forearm).
# Doing Z_score for each subject and channel separately, ends up with nan in the ch9 and ch10 of this rwo subjects.
# 2) -> In DB3 (for some reason subject 1 has done less exercise) -> Ex done = [ 0,  4,  7,  8,  9, 10, 11, 12, 13]
# It doesn't seem a wrong labeling, because the numbers for each gesture are within the normal range, but I decided to drop him.
# He corresponds to subject 63 in the merged DB2+DB7+DB3
df = df.fillna(0)

#%% Exluding from df subject that will be used for test
if DATABASE == 'DB3':
    df = df.query('sub != 1')
    left_hand_DB3 = [2, 5, 6]
    df.drop(df[df['sub'].isin(left_hand_DB3)].index, inplace=True)
elif DATABASE == 'DB2+DB7+DB3':     # here i don't need to drop left-handed, because i did it a creation of dataframe
    df = df.query('sub != 63')
    # df = df.query('sub != 66')       # Enourmus dash score, already checked bad performance in all cases.

# Select sub to test
random.seed(32)
sub_to_test = random.sample(list(df['sub'].unique()), k= round(df['sub'].nunique()/5))  # for TL framework

if DATABASE == 'DB2+DB7+DB3':
    sub_to_test.append(66)
    sub_to_test.append(69)
    sub_to_test.append(71)  # Added for testing
# [42, 6, 28, 47, 66, 70, 21, 7, 38, 71, 31, 32]    -> 12 DB2+DB7+DB3 -> 3 from DB3 seed 29 [71 = 9, 70 = 8, 66 = 4]
# [6, 16, 11, 23, 53, 18, 37, 2, 56, 3, 8, 24]    -> 12 DB2+DB7+DB3 -> 1 from DB3 seed 32 
# [6, 28, 21, 7, 38]    -> 5 subject in DB2
# sub_to_test_df = df[df['sub'].isin(sub_to_test)]

sub_to_test_df = df[df['sub'].isin(sub_to_test)]
# Drop subject to test and to apply TL after training
df = df.loc[~(df['sub'].isin(sub_to_test))]

# %% Fixing and ordering "sub" column numbers [from 0 to X] for domain classifier.
if TRAIN_TYPE == 'Reversal' or 'TCN_SAM_CAM':
    
    filename = filename + f"_lamba_{GAMMA}"
    # Create 5 groups for lowering Domain classifier burden & exploit domain loss during validation
    if GROUPS is True:
        # shuffle subjects before to fit them in a group
        n_subject2 = np.unique(df['sub'])
        df['group'] = np.zeros(len(df)).astype(np.uint8)    # Create new column
        group_size = len(n_subject2) // 5
        remainder = len(n_subject2) % 5
        # Attention! Because both "groups" column and train_val_test_split_df works similar way, seeds here MUST be different from 32
        np.random.seed(24)  
        np.random.shuffle(n_subject2)
        # fill 'group' column with same value for each 10 subjects to create 5 domain 
        for i in range(5):
            mask = df['sub'].isin(n_subject2[i * group_size: (i + 1) * group_size])
            df.loc[mask, 'group'] = i
        # Allocate each subject not part of a group in different groups
        if remainder != 0:
            print(f'Division in groups not integer! redistributing {remainder} subjects')
            for i in range(remainder):
                df.loc[df['sub'] == n_subject2[- (i+1)], 'group'] = i
            
        print(f' Number of samples per Group:\n{df["group"].value_counts()}')
        # df['sub'] = df['group']       
        # This line has been moved AFTER the splitting so that all 5 groups are likely present in both valid and train

        filename = filename + '_5_Groups'
    
    elif AMPUTEE is True:  # Create new column 
         # Sub 61 & 62 are subs 21 & 22 from DB7, while the others are from DB3
        amputee_subs = df['sub'].unique()[-6:]  
        df['amputee'] = np.where(df['sub'].isin(amputee_subs), 1, 0)
        
        filename = filename + f'_Amputee_lamba_{GAMMA}'

    else:    # Domain classifies over all subjects
        
        counts = df['sub'].value_counts(sort=False)
        
        # Create the ordered vector using the counts of unique values
        ordered_sub = np.concatenate([np.full(count, i) for i, count in enumerate(counts)])
        # Rewriting df['sub'] to be from 0 to n_sub (in order to classify the domain with cross-entropy loss)
        df['sub'] = ordered_sub.astype(np.uint8)

    # Reset index
    df = df.reset_index(drop=True)
    df = df.set_index(pd.RangeIndex(start=0, stop=len(df)))
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    
    if DOM_VAL_LOSS:
        filename = filename + '_dom_val_loss'


#%% Split dataframe
# Splitting Train, val and test based on Rep
n_subject, n_rep = np.unique(df['sub']), np.unique(df['rep'])

if SPLIT == 'rep':
    train_val_test_rep = [[2, 4, 6], [1, 5], [3]]       # [[1, 2, 5, 6], [4], [3]]
    df_train, df_val, df_test = train_val_test_split_df(df, mode=SPLIT, manual_sel=train_val_test_rep, seed = 32)
    
    valid_group = df_val[SPLIT].unique()
    test_group = df_test[SPLIT].unique()[0]  # Usually only one rep is used for testing, modify it accordingly
    
    check_dom_loss = True
    
    # Swap 'sub' column with amputee column to perform Adversarial Network Healty vs Amputee
    if AMPUTEE:
        df_train['sub'], df_val['sub'], df_test['sub'] = df_train['amputee'], df_val['amputee'], df_test['amputee']
        del df_train['amputee'], df_val['amputee'], df_test['amputee']
    
    # Swap 'sub' column with different casually selcted groups to perform adversarial with less domain (5 vs 60, easier loss for domain classifier)  
    elif GROUPS:
        print(f'Number of samples per Group after splitting: \nTRAIN: \n {df_train["group"].value_counts()}\n\n\
                VALID:\n {df_val["group"].value_counts()} ')
            
        df_train.loc[:, 'sub'], df_val.loc[:, 'sub'] = df_train['group'], df_val['group']
        del df_train['group'], df_val['group'], df_test['Unnamed: 0']
        
# Splitting Train and val based on Sub (In this case, considering a cross-subject experiment, the test will be "sub_to_test")
elif SPLIT == 'sub':  # subject 5->175[cm]x75[kg] good tester in average
    if AMPUTEE:
        np.random.seed(32)
        # Taking 6 casual healty subjcets: ~1/6 (39 healty) // [39, 12, 40, 34, 15, 42]
        valid_group = np.random.choice(df['sub'].unique()[:-7], size = 6)   
        valid_group =  np.append(valid_group, [73])     # 90% forearm remianing, low DASH, 5 years Myoelectric usage

        df_train = df[~df['sub'].isin(valid_group)].copy()
        df_val = df[df['sub'].isin(valid_group)].copy()
        df_test = sub_to_test_df
        test_group = df_test['sub'].unique()
        # Swap 'sub' & 'amputee' columns for domain classifier and delete useless column 'amputee'
        df_train['sub'], df_val['sub'] = df_train['amputee'], df_val['amputee']
        
        # You can in this specific case of cross subject training check how domain calssifier performs over domain
        check_dom_loss = True
        del df_train['amputee'], df_val['amputee']
        
    elif GROUPS:           
            # Get unique subjects in the first group to determine size for validation
            first_group_subjects = df[df['group'] == 0]['sub'].unique()  # Assuming group 0 is the first group
            
            val_subjects = []
            for group in df['group'].unique():
                # Get unique subjects from the current group
                group_subjects = df[df['group'] == group]['sub'].unique()
                
                # Randomly select two subjects from the current group
                selected_subjects = np.random.choice(group_subjects, size= round(len(first_group_subjects) / 5), replace=False)
                
                # Append the selected subjects to the validation subjects list
                val_subjects.extend(selected_subjects)

            # Create the validation DataFrame (df_val) by filtering df for the selected subjects
            df_val = df[df['sub'].isin(val_subjects)]

            # Create the train DataFrame (df_train) by excluding the selected validation subjects
            df_train = df[~df['sub'].isin(val_subjects)]

            # Manually assign the test set from a separate DataFrame (e.g., sub_to_test_df)
            df_test = sub_to_test_df

            # After the manual splitting, we can print the counts per group for the train and validation sets
            valid_group = df_val['sub'].unique()
            test_group = df_test['sub'].unique()

            print(f'Number of samples per Group after splitting: \n \
                TRAIN: \n {df_train["group"].value_counts()}\n\n \
                VALID:\n {df_val["group"].value_counts()} ')
            
            df_train.loc[:, 'sub'], df_val.loc[:, 'sub'] = df_train['group'], df_val['group']
            del df_train['group'], df_val['group']
            check_dom_loss = True
        
    else:
        # Swapping test and validation order for how the function works 
        df_train, df_test, df_val = train_val_test_split_df(df, mode=SPLIT, percentages = [0.8, 0.0, 0.20], seed = 34)
        # In cross-subject experiment, in order to maximize variability and subs employed for training/validation,
        # I directly test the model over the subjects originally left out for TL, as they are unseen subjects as well.
        df_test = sub_to_test_df
        
        valid_group = df_val['sub'].unique()
        test_group = df_test['sub'].unique()
        # Used to avoid problems during evaluation in training when df_train['sub'].unique() != df_val['sub'].unique()
        # so domain classifier doesn't tilt in evaluation mode
        check_dom_loss = False
        # Swap columns for reversal
        
        counts = df_train['sub'].value_counts(sort=False)
        # Create the ordered vector using the counts of unique values
        ordered_sub = np.concatenate([np.full(count, i) for i, count in enumerate(counts)])
        # Rewriting df_train['sub'] to be from 0 to n_sub (in order to classify the domain with cross-entropy loss) (no need for validation as it doesn't use reversal)
        df_train.loc[:, 'sub'] = ordered_sub.astype(np.uint8)

# Reset index
df_train = df_train.reset_index(drop=True)
df_train = df_train.set_index(pd.RangeIndex(start=0, stop=len(df_train)))
df_train = df_train.drop(columns=['Unnamed: 0'], errors='ignore')
    
del df

# %% DATASETS
# You must pass to the dataloader a groupby dataframe
if TRAIN_TYPE == 'Standard':      # 2 output
    train_set = Pandas_Dataset(df_train.groupby('sample_index'))
    valid_set = Pandas_Dataset(df_val.groupby('sample_index'))

elif TRAIN_TYPE == 'Reversal' or TRAIN_TYPE == 'TCN_SAM_CAM':          # 3 output
    train_set = Pandas_Dataset(df_train.groupby('sample_index'), target_col='sub')
    valid_set = Pandas_Dataset(df_val.groupby('sample_index'), target_col='sub')

elif TRAIN_TYPE == 'Triplet' or  TRAIN_TYPE == 'Pre_Triplet':        # 4 output
    train_set = dataframe_dataset_triplet(df_train, groupby_col='sample_index', target_col='sub')
    valid_set = dataframe_dataset_triplet(df_val, groupby_col='sample_index', target_col='sub')

elif TRAIN_TYPE == 'JS':        # 3 output
    train_set = dataframe_dataset_JS(df_train, groupby_col='sample_index', target_col='sub')
    valid_set = dataframe_dataset_JS(df_val, groupby_col='sample_index', target_col='sub')
    # The following variable is changed because JS works along with centre loss, which calculate centers of embedding
    # with respect to previous iteration. If batch changes it raises error for last batch-in-epoch.
    drop_last = True
else:
    print('Training type does not match! Choose between:\n"Standard"\n"Reversal"\n"Triplet"\n"JS"\n"TCN_SAM_CAM"')

# For the test I just need to check metrics, no need for more slow and complex dataloaders
test_set = Pandas_Dataset(df_test.groupby('sample_index'))

# %% Dataloaders
batch_size = 256
num_workers = 0
if 'drop_last' not in  locals():
    drop_last = False

params = {'batch_size': batch_size,
          'shuffle': True,
          # 'sampler': sampler,
          'num_workers': num_workers,
          'drop_last': drop_last}

train_set_generator = data.DataLoader(train_set, **params)
valid_set_generator = data.DataLoader(valid_set, **params)
test_set_generator = data.DataLoader(test_set, **params)

# %% Model parameters
# Default
# in_ch = 1
# out_ch = np.array([32, 128, 128, 128])
# kernel_sizes = np.full((5, 2), [1, 3])
# for i in range(5):
#     kernel_sizes[i][0] = 10 * (i + 1)
# kernel_dim = np.array([1, 3, 3])
# number_of_classes = 14

# Grid
# Creating kernels on the desired window length: you can change as you wish, as the model is fully parametrized.
kernels_gap = [g for g in range(0, 3 * round(WND_LEN / 20), round(WND_LEN / 20))]
kernel_sizes = np.full((3, 5, 2), [1, 3])
for j in range(3):
    for i in range(5):
        kernel_sizes[j][i][0] = (round(WND_LEN / 20) * (i + 1) + kernels_gap[j])
n_classes = 14

if ACT_FUN == 'relu':
    ACT_FUN = nn.ReLU()
elif ACT_FUN == 'elu':
    ACT_FUN = nn.ELU()
elif ACT_FUN == 'prelu':
    ACT_FUN = nn.PReLU()
elif ACT_FUN == 'lrelu':
    ACT_FUN = nn.LeakyReLU()


net= {'N_multik': 32, 'N_Conv_conc': 128, 'N_SepConv': 256,
                'Kernel_multi_dim': kernel_sizes[0],
                'Kernel_Conv_conc': 1,
                'act_func':ACT_FUN,
                'Pool_Type': nn.MaxPool2d,
                'wnd_len': WND_LEN,
                 # Key used only for reversal gradient
                'Output2': nn.Flatten  # -> the domain classification will start at the first linear
                # 'Output2': MultiKernelConv2D_grid  # -> the domain classification will start after the multikernel stage
                }

# %% Grid initialization
if TRAIN_TYPE == 'Standard' or TRAIN_TYPE == 'Triplet' or  TRAIN_TYPE == 'Pre_Triplet' or TRAIN_TYPE == 'JS':
    model = MKCNN_grid(net)

elif TRAIN_TYPE == 'Reversal':
    model = MKCNN_grid_AN(net, num_domains=len(np.unique(df_train['sub'])))
    # model = MKCNN_grid_ATTN(net, num_domains=len(np.unique(df_train['sub'])))
    
elif TRAIN_TYPE == 'TCN_SAM_CAM':
    # l2_factor = 0.2
    model = TCN_ch_sp_ATT(wnd_len=WND_LEN, n_classes=n_classes,num_domains= len(np.unique(df_train['sub'])), dom_out=net['Output2'])

model = model.to(device)

total_params= sum([p.numel() for p in model.parameters() if p.requires_grad])
print(f'Model created with Num params: {total_params}')
# %% Loss Optim and scheduler
# Define Loss functions
cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean').to(device)
# triplet = nn.TripletMarginLoss(reduction='mean', margin=1, p=2)

# Define Optimizer
learning_rate = 0.0005
# changed beta values from (0.5,0.999) to (0.9,0.999)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# # Define Scheduler
precision = 1e-6
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=.2,  patience=5, verbose=True, eps=precision)


# %% Training and validation
num_epochs = EPOCHS
if TRAIN_TYPE == 'Standard':
    best_weights, tr_losses, val_losses = train_model_standard(model=model, loss_fun=cross_entropy_loss,
                                                               optimizer=optimizer, scheduler=scheduler,
                                                               dataloaders={"train": train_set_generator,
                                                                            "val": valid_set_generator},
                                                               num_epochs=num_epochs, precision=precision,
                                                               patience=10, patience_increase=8, device=device)
elif TRAIN_TYPE == 'Triplet':
    best_weights, tr_losses, val_losses = train_model_triplet(model=model, loss_fun=cross_entropy_loss,
                                                              optimizer=optimizer, scheduler=scheduler,
                                                              dataloaders={"train": train_set_generator,
                                                                           "val": valid_set_generator},
                                                              num_epochs=int(num_epochs/2), precision=precision,
                                                              patience=10, patience_increase=8, beta=0.5, device=device)
elif TRAIN_TYPE == 'Pre_Triplet': 
    best_weights, tr_losses, val_losses = pre_train_model_triplet(model=model, 
                                                              loss_fun= nn.TripletMarginLoss(reduction='mean', swap=True),
                                                              optimizer=optimizer, scheduler=scheduler,
                                                              dataloaders={"train": train_set_generator,
                                                                           "val": valid_set_generator},
                                                              num_epochs=int(num_epochs), precision=precision,
                                                              patience=10, patience_increase=8, device=device)

elif TRAIN_TYPE == 'Reversal' or  TRAIN_TYPE == 'TCN_SAM_CAM':
    best_weights, tr_losses, val_losses, tr_dom_losses, val_dom_losses, tr_task_losses, val_task_losses = \
        train_model_reversal_gradient(model=model, loss_fun=cross_entropy_loss, loss_fun_domain=cross_entropy_loss,
                                      lamba=GAMMA, optimizer=optimizer, scheduler=scheduler,
                                      dataloaders={"train": train_set_generator, "val": valid_set_generator},
                                      num_epochs=num_epochs , precision=precision, patience=10, patience_increase=15, device = device,
                                      combined_val_loss =  DOM_VAL_LOSS, check_val_dom_loss = check_dom_loss)
                                      # More patient increase because at the beginning it starts to overfit

elif TRAIN_TYPE == 'JS':
    best_weights, \
        tr_losses, val_losses, \
        tr_dom_losses, val_dom_losses, \
        tr_center_losses, val_center_losses, \
        tr_task_losses, val_task_losses = \
        train_model_JS(model=model, loss_fun=cross_entropy_loss, lambda1=0.5, lambda2=0.5,
                            optimizer=optimizer, scheduler=scheduler,
                            dataloaders={"train": train_set_generator, "val": valid_set_generator},
                            num_epochs=int(num_epochs/2), precision=precision, patience=10, patience_increase=8)
    

# %% Savings
path = database + f'/{TRAIN_TYPE}/Cross_{SPLIT}'
if not os.path.exists(path):
    os.makedirs(path)

# if TRAIN_TYPE == 'Reversal':   # Output 2 Refers to the starting head of the second classifier (subject classifier)
#     if net['Output2'] == MultiKernelConv2D_grid:
#         filename = filename + '_middle'
#     elif net['Output2'] == nn.Flatten:
#         filename = filename + '_last'


# Save state dict of the model
if not os.path.exists(path + f'/Best_States/'):
    os.makedirs(path + f'/Best_States/')
torch.save(best_weights['state_dict'],path + f'/Best_States/state_dict_{filename}.pth')


# %% PlotLoss
if not os.path.exists(path + '/Plot/'):
    os.makedirs(path + '/Plot/')

PlotLoss(tr_losses, val_loss=val_losses,
         title=f'Cross-{SPLIT} on {SPLIT} {test_group}',
         path_to_save=path + '/Plot/',
         filename=f'Cross_{SPLIT}_{filename}.png')

# For reversal gradient and JS + center loss
if TRAIN_TYPE == 'Reversal' or TRAIN_TYPE == 'JS' or  TRAIN_TYPE == 'TCN_SAM_CAM':
    if check_dom_loss:
        PlotLoss(tr_dom_losses, val_loss=val_dom_losses,
                title=f'Cross-{SPLIT} on {SPLIT} {test_group}',
                path_to_save=path + '/Plot/',
                filename=f'Cross_{SPLIT}_{filename}_DOMAIN_ONLY.png')
    if DOM_VAL_LOSS:
        PlotLoss(tr_task_losses, val_loss=val_task_losses,
                title=f'Cross-{SPLIT} on {SPLIT} {test_group}',
                path_to_save=path + '/Plot/',
                filename=f'Cross_{SPLIT}_{filename}_TASK_ONLY.png')

if 'tr_center_losses' in locals():
    # Only for JS + Center Loss
    PlotLoss(tr_center_losses, val_loss=val_center_losses,
             title=f'Cross-{SPLIT} on {SPLIT} {test_group}',
             path_to_save=path + '/Plot/',
             filename=f'Cross_{SPLIT}_{filename}_Center_Loss.png')

# %% Conf Matrix
if not os.path.exists(path + '/Conf_Matrix/'):
    os.makedirs(path + '/Conf_Matrix/')

# Evaluation
softmax_block = nn.Softmax(dim=1)
y_true = []
y_pred = []


model.eval()
with torch.no_grad():
    for inputs, labels in test_set_generator:   
        # inputs = torch.swapaxes(inputs, 2, 1)  # -> convert from [10,20] to [20,10] for DB1
        inputs = inputs[:, None, :, :]
        inputs = inputs.to(device)
        labels_np = labels.cpu().data.numpy()
        # forward
        outputs, _ = model(inputs)
        outputs_np = softmax_block(outputs)
        outputs_np = outputs_np.cpu().data.numpy()
        outputs_np = np.argmax(outputs_np, axis=1)

        y_pred = np.append(y_pred, outputs_np)
        y_true = np.append(y_true, labels_np)

    cm = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
    # Fancy confusion matrix
    plot_confusion_matrix(cm, target_names=exe_labels, title=f'Confusion Matrix for {SPLIT} {test_group}',
                          path_to_save=path + f'/Conf_Matrix/{filename}.png')

# Checking time of inferenze of ìììaveraged over 100 samples (batch_size = 1 to replicate real scenario)
time_tot = 0
time_dataloader = data.DataLoader(test_set, batch_size=1, shuffle=True)
iterable_dataloader = iter(time_dataloader)
for t in range(100):
    j, _ = next(iterable_dataloader)
    j = j[:, None, :, :].to(device)
    start_time = time.time()
    _ = model(j)
    time_tot = time_tot + (time.time() - start_time)
avg_time = (time_tot / 100) * 1000

del df_test, test_set, test_set_generator

# %% Write in csv cross_sub results
# Building columns
header_net = ['Name', f'Tested_{SPLIT}', 'Best Val Loss', 'Accuracy', 'Kappa', 'F1_score', 'Best Epoch',
              'Average Inference Time [ms]', 'Norm_Type', 'Norm_Mode', 'Rect', f'Valid_{SPLIT}']

# Open the CSV file and write the headers and row of values
with open(path + f'/Evals.csv', 'a', newline='') as myFile:
    writer = csv.writer(myFile)
    if myFile.tell() == 0:
        writer.writerow(header_net)
    # Create the row of values
    row = [filename, test_group, min(val_losses),
           metrics.accuracy_score(y_true=y_true, y_pred=y_pred),
           metrics.cohen_kappa_score(y1=y_true, y2=y_pred, weights='quadratic'),
           metrics.f1_score(y_true=y_true, y_pred=y_pred, average='weighted'),
           best_weights['epoch'], avg_time,
           NORM_TYPE, NORM_MODE, str(RECT), valid_group]
    writer.writerow(row)
print(f'Results Saved in -> {path}/Evals.csv')

# Check results
# myFile = open(database + f'Cross_sub/Best_States/Sub_{sub}.csv', 'r')
# print("The content of the csv file is:")
# print(myFile.read())
# myFile.close()


####################################################################################
####################################################################################
####################################################################################
####################################################################################


# %% Freezing params:
model.load_state_dict(torch.load(f'{database}/{TRAIN_TYPE}/Cross_{SPLIT}/Best_States/state_dict_{filename}.pth'))
# Freezing parameters
if TRAIN_TYPE != 'TCN_SAM_CAM':
    if FREEZE == 'last':
        # Freeze first part
        model.freeze_multik()   
        batch_size = 64
        # Leaving unfrozen only last 2 Linear Layers
        for layer in model.model:       
            if not isinstance(layer, nn.Linear):
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                break

    elif FREEZE == 'middle':
        batch_size = 64
        # Freeze first part only
        model.freeze_multik()
        
    if TRAIN_TYPE == 'Reversal':
        for layer in model.domain_classifier:
            for param in layer.parameters():
                param.requires_grad = False

# Only for TCN_SAM_CAM because it is different
else:
    batch_size = 64
    # Freeze all
    for param in model.parameters():        
        param.requires_grad = False
    # Unfreeze last
    for layer in model.conv_block:
        # Unfreeze last linear layers
        if isinstance(layer, nn.Linear): 
            for param in layer.parameters():
                param.requires_grad = True



num_param_tl = sum([p.numel() for p in model.parameters() if p.requires_grad])
print(f'Freezing {total_params - num_param_tl} parameters -> TL model has now {num_param_tl} trainable parameters')

# %% Param for TL dataloader
num_workers = 1
params = {'batch_size': batch_size,
        'shuffle': True,
        # 'sampler': sampler,
        'num_workers': num_workers,
        'drop_last': False}


# Parameters for optimizer and scheduler
lr = 0.01
betas = (0.9, 0.999)
weight_decay = 1e-3
scheduler_factor = 0.2
scheduler_patience = 5

scheduler_eps = 1e-6

#%% Cycle over sub_to_test and over reps
# -> Here it is possible to change the training type for fitting new user.
# For example, you can apply Adversarial Network with reversal gradient to learn Repetition-invariant-feature for the new subject.
cc = 1
n_rep = [2, 4, 1, 5, 6]
# rep 3 is used to test
test_rep = [3]

initial_state = copy.deepcopy(model.state_dict())
# if TRAIN_TYPE == 'Standard' or  TRAIN_TYPE == 'Reversal':
for subx in sub_to_test[2:]:
    print(70 * '#', 5 * '\n', f'Testing sub Number: {subx} -> {cc} / {len(sub_to_test)}', 5 * '\n', 70 * '#')
    df = sub_to_test_df[sub_to_test_df['sub'] == subx]
    cc = cc + 1
    for n_rep_tr in range(len(n_rep)):      # Varying number of repetition used to train
        for val_rep in n_rep:
            val_rep = [val_rep]     # I need a list for query the dataframe
            n_rep_copy = n_rep.copy()
            n_rep_copy.remove(val_rep[0])

            idx = n_rep.index(val_rep[0])
            tr_reps = [n_rep_copy[(idx + i) % len(n_rep_copy)] for i in range(n_rep_tr)]
            print(70 * '#', 3 * '\n',\
                f'Testing sub Number: {cc}/{len(sub_to_test)}',"    Num train rep:", n_rep_tr, "     Train Reps:", tr_reps, f'   Val Rep: {val_rep}', \
                    3 * '\n', 70 * '#')
            #combinations = itertools.combinations(n_rep_copy, n_rep_tr)
            # for tr_reps in combinations:
            #     print("Cycle", n_rep_tr, ":", list(tr_reps), "Val rep:", val_rep)
            
            # Reload the model's initial state
            model.load_state_dict(initial_state)
            
            # # Re-inituialize lienar layer, responsible for classification only
            # model.reinitialize_last_layers()
            
            # Reinitialize the optimizer with the same parameters
            optimizer_tl = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

            # Reinitialize the scheduler with the new optimizer
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_tl, factor=scheduler_factor,
                                                                    patience=scheduler_patience, eps=scheduler_eps)
            # Reinitialize the loss function
            cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean').to(device)
            
            get_TL_results(model=model,
                        tested_sub_df=df, train_type='Standard',
                        train_rep=tr_reps, valid_rep = val_rep, test_rep = test_rep,
                        exe_labels=exe_labels,
                        num_epochs=15,
                        loss_fn=cross_entropy_loss, optimizer=optimizer_tl, **params, scheduler=scheduler, device=device,

                        filename=None, path_to_save=f'{database}/{TRAIN_TYPE}/Cross_{SPLIT}/TL/{FREEZE}/{filename}_2/')

####################################################################################
####################################################################################
####################################################################################
####################################################################################
# %%
# Averaging
# Read the CSV file into a DataFrame
# for TRAIN_TYPE in ['Standard', 'Pre_Triplet', 'Reversal']:
#     if TRAIN_TYPE == 'Reversal':    # amputee and non amputee
#         for i in range(2):
#             if i==0:                
#                 filename = filename + '_lamba_0.9' # No amputee
#             elif i ==1:
#                 filename = filename[0:-10] + '_Amputee_lamba_0.9'
                
#     for SPLIT in ['rep', 'sub']:
#         for FREEZE in ['last','middle']:
evals = pd.read_csv(f'{database}/{TRAIN_TYPE}/Cross_{SPLIT}/TL/{FREEZE}/{filename}/Evals.csv')
# evals = pd.read_csv('/home/riccardo/EMG_data/DB2+DB7+DB3/Standard/TL/last/Evals.csv')

# Drop subjects linked to previous runs or seeds
if DATABASE == 'DB2+DB7+DB3':
    correct_subs = [11, 16, 18, 23, 24,  2, 37,  3, 53, 56, 66, 69,  6, 71,  8]     # Extracted wwith seed
    evals = evals[evals['Tested_sub'].isin(correct_subs)]

# Sort the DataFrame by the desired column
evals = evals.sort_values(by='Name')

# Exclude columns from the header
exclude_columns = ["Train Repetitions", "Val Repetition", "Num trainable Param"]
# Prepare next csv file
evals_avg = pd.DataFrame(columns=[col for col in evals.columns if col not in exclude_columns])
std_avg = pd.DataFrame(columns=[col for col in evals.columns if col not in exclude_columns])

if len(evals) % 5 != 0:
    raise ValueError("The number of rows in 'evals' is not divisible by 5")
# Calculate the number of groups of 5 rows each (k-fold repetition validation)
num_groups = len(evals) // 5

# Iterate through each group of 5 rows
for i in range(num_groups):
    # Calculate the start and end index for each group of 5 rows
    start_idx = i * 5
    end_idx = start_idx + 5

    # Get the subset of rows for each group
    group_df = evals.iloc[start_idx:end_idx]

    # Set the value of "Name" and "Tested_sub" for the group
    evals_avg.loc[i, "Name"] = group_df.iloc[0]["Name"]
    evals_avg.loc[i, "Tested_sub"] = group_df.iloc[0]["Tested_sub"]
    std_avg.loc[i, "Name"] = group_df.iloc[0]["Name"]
    std_avg.loc[i, "Tested_sub"] = group_df.iloc[0]["Tested_sub"]

    # Calculate the average of each desired column for the group
    avg_values = group_df[["Best Val Loss", "Accuracy", "Kappa", "F1_score", "Best Epoch"]].mean()
    std_values = group_df[["Best Val Loss", "Accuracy", "Kappa", "F1_score", "Best Epoch"]].std()

    # Update the average values in the new dataframe
    evals_avg.loc[i, ["Best Val Loss", "Accuracy", "Kappa", "F1_score", "Best Epoch"]] = avg_values
    std_avg.loc[i, ["Best Val Loss", "Accuracy", "Kappa", "F1_score", "Best Epoch"]] = std_values
std_avg.fillna(0, inplace = True)

# Write the new dataframe to a CSV file

evals_avg.to_csv(f'{database}/{TRAIN_TYPE}/Cross_{SPLIT}/TL/{FREEZE}/{filename}/evals_avg.csv', index = False)
std_avg.to_csv(f'{database}/{TRAIN_TYPE}/Cross_{SPLIT}/TL/{FREEZE}/{filename}/std_avg.csv', index = False)

# %% 
# Testing
import matplotlib.pyplot as plt
# Try drop 3 subject from DB3, e.g. 66,69 71
evals_avg_healty = evals_avg[~evals_avg['Tested_sub'].isin([66,69,71])]
std_avg_healty = std_avg[~std_avg['Tested_sub'].isin([66,69,71])]

# Group by index modulo 5 and calculate the mean of the "F1_score" column
grouped_avg = evals_avg.groupby(evals_avg.index % 5)["Accuracy"].agg(['mean', 'std'])
grouped_avg_healty = evals_avg_healty.groupby(evals_avg_healty.index % 5)["Accuracy"].agg(['mean', 'std'])

# grouped_std = std_avg.groupby(std_avg.index % 5)["F1_score"]   
# grouped_std_healty = std_avg_healty.groupby(std_avg_healty.index % 5)["F1_score"]
# combined_std = []
# combined_std_healty = []

# for i in range(5):   # number of X points e.g. number of repetitions used for training
#     group = np.array(grouped_std.get_group(i)) # Taking group of subs trained with i reps
#     consum = 0
#     group_healty = np.array(grouped_std_healty.get_group(i))
#     consum_healty = 0
#     for j in range(len(group)): # for each subs calculate combined
#         tmp = group[j]**2 + (evals_avg.loc[5*j + 4, 'F1_score'] - grouped_avg[i]) **2
#         tmp_healty = group[j]**2 + (evals_avg.loc[5*j + 4, 'F1_score'] - grouped_avg[i]) **2
#         consum =+ tmp
#         consum_healty =+ tmp
        
#     combined_std.append(consum)
#     combined_std_healty.append(consum_healty)

# Plot the mean F1_score values against the repetitions used
plt.plot(range(5), grouped_avg['mean'], marker='o', markersize=4, label='Mean Accuracy', color = 'red', linewidth = 1.3)
plt.errorbar(range(5), grouped_avg['mean'], yerr=grouped_avg['std'], fmt='none', capsize=5, color='red', linewidth = 1.3)
 # Those 3 amputee are possible to plot opnly in the combined database
if DATABASE == 'DB2+DB7+DB3':
    plt.plot(range(5), grouped_avg_healty['mean'], marker='o', markersize=4, label='Mean Accuracy', color = 'blue', linewidth = 1.3)
    plt.errorbar(range(5), grouped_avg_healty['mean'], yerr=grouped_avg_healty['std'], fmt='none', capsize=5, color='blue', linewidth = 1.3)
    plt.legend(labels=['All Subjects', 'Healthy Subjects'], loc='upper left')

best_epoch_means = evals_avg.groupby(evals_avg.index % 5)["Best Epoch"].mean()
for i in range(1, 5):  # Exclude the first value (index 0) in the x-axis
    # Determine the position for the text box (just above the x-axis line)
    x_pos = i- 0.25
    y_pos = plt.gca().get_ylim()[0]  # Get the lower limit of the y-axis
    text_offset = 0.06  # Offset factor to adjust the vertical position of the text box
    
    # Add the text box displaying the average "Best Epoch" value
    plt.text(x_pos, y_pos + text_offset, f"Mean\nEpochs: {best_epoch_means[i]:.2f}", ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))
# Boxplot
# plt.boxplot([evals_avg.loc[evals_avg.index % 5 == i, 'F1_score'] for i in range(5)], positions=range(5), widths=0.5)

plt.xlabel('Number of Repetitions used for training')
# plt.title('Mean F1_score over 15 (3A) subjects')
plt.ylabel('Accuracy ± Std')
plt.xticks(range(5))
plt.grid(True)
# plt.show()
plt.savefig(f'{database}/{TRAIN_TYPE}/Cross_{SPLIT}/TL/{FREEZE}/{filename}/plot.png')
plt.close()


# %% Confront AMPUTEEES
# import matplotlib.pyplot as plt
# database = f'/home/riccardo/EMG_data/{DATABASE}/'
# filename = f'wnd_{WND_LEN}_{NORM_TYPE}_{NORM_MODE}_rect_{RECT}'

# filename = filename + '_Amputee'
# filename = filename + f'_lamba_{GAMMA}'

# filename = filename[8:]
# evals_200 = pd.read_csv(f'{database}/{TRAIN_TYPE}/Cross_{SPLIT}/TL/{FREEZE}/wnd_200_{filename}/Evals.csv')
# evals_300 = pd.read_csv(f'{database}{TRAIN_TYPE}/Cross_{SPLIT}/TL/{FREEZE}/wnd_300_{filename}/Evals.csv')
# evals_200 = evals_200.sort_values(by='Name')
# evals_300 = evals_200.sort_values(by='Name')
# evals_avg_200 = pd.read_csv(f"{database}/{TRAIN_TYPE}/Cross_{SPLIT}/TL/{FREEZE}/wnd_200_{filename}/evals_avg.csv")
# evals_avg_300 = pd.read_csv(f"{database}/{TRAIN_TYPE}/Cross_{SPLIT}/TL/{FREEZE}/wnd_300_{filename}/evals_avg.csv")

# if DATABASE == 'DB2+DB7+DB3':
#     # Filter the data to include only subjects 66, 69, and 71 --> = subs 4,7 and 9 of DB3
#     evals_avg_66_200 = evals_avg_200[evals_avg_200['Tested_sub'] == 66]
#     evals_avg_69_200 = evals_avg_200[evals_avg_200['Tested_sub'] == 69]
#     evals_avg_71_200 = evals_avg_200[evals_avg_200['Tested_sub'] == 71]
    
#     evals_avg_66_300 = evals_avg_300[evals_avg_300['Tested_sub'] == 66]
#     evals_avg_69_300 = evals_avg_300[evals_avg_300['Tested_sub'] == 69]
#     evals_avg_71_300 = evals_avg_300[evals_avg_300['Tested_sub'] == 71]
    
#     # Filter the data to include only subjects 66, 69, and 71
#     evals_amputee_200 = evals_200[evals_200['Tested_sub'].isin([66, 69, 71])]
#     std_values_dict_200 = {}
#     evals_amputee_300 = evals_300[evals_300['Tested_sub'].isin([66, 69, 71])]
#     std_values_dict_300 = {}

#     for subs in [66, 69, 71]:
#         eval_sub_200 = evals_200[evals_200['Tested_sub'] == subs] 
#         eval_sub_300 = evals_300[evals_300['Tested_sub'] == subs] 
#         # std_values = eval_sub.groupby(eval_sub.index % 5)["F1_score"].std()
#         std_values_200 = [eval_sub_200['Accuracy'].iloc[i:i+5].std() for i in range(0, len(eval_sub_200), 5)][:5]
#         std_values_300 = [eval_sub_300['Accuracy'].iloc[i:i+5].std() for i in range(0, len(eval_sub_300), 5)][:5]
#         std_values_dict_200[subs] = std_values_200
#         std_values_dict_300[subs] = std_values_300
        

#     # Plot Mean Accuracy values against the repetitions for each subject
#     plt.plot(range(5), evals_avg_66_200['Accuracy'], marker='o', markersize=4, label='Subject 4', color='blue', linewidth=1.3) 
#     plt.plot(range(5), evals_avg_69_200['Accuracy'], marker='o', markersize=4, label='Subject 7', color='green', linewidth=1.3)
#     plt.plot(range(5), evals_avg_71_200['Accuracy'], marker='o', markersize=4, label='Subject 9', color='red', linewidth=1.3) 
    
#     plt.plot(range(5), evals_avg_66_300['Accuracy'], marker='o', markersize=4, linestyle='--', color='blue', linewidth=1.3) 
#     plt.plot(range(5), evals_avg_69_300['Accuracy'], marker='o', markersize=4, linestyle='--', color='green', linewidth=1.3) 
#     plt.plot(range(5), evals_avg_71_300['Accuracy'], marker='o', markersize=4, linestyle='--', color='red', linewidth=1.3)
    
#     errorbar_xvals = [1, 2, 3, 4]   # To not plot errorbar for std = 0 when no retrain performed
    
#     plt.errorbar(errorbar_xvals, evals_avg_66_200['Accuracy'].iloc[errorbar_xvals], yerr=[std_values_dict_200[66][i] for i in errorbar_xvals], fmt='none', ecolor='blue', capsize=5, linewidth=1.3)
#     plt.errorbar(errorbar_xvals, evals_avg_69_200['Accuracy'].iloc[errorbar_xvals], yerr=[std_values_dict_200[69][i] for i in errorbar_xvals], fmt='none', ecolor='green', capsize=5, linewidth=1.3)
#     plt.errorbar(errorbar_xvals, evals_avg_71_200['Accuracy'].iloc[errorbar_xvals], yerr=[std_values_dict_200[71][i] for i in errorbar_xvals], fmt='none', ecolor='red', capsize=5, linewidth=1.3)
    
#     plt.errorbar(errorbar_xvals, evals_avg_66_300['Accuracy'].iloc[errorbar_xvals], yerr=[std_values_dict_300[66][i] for i in errorbar_xvals], fmt='none', ecolor='blue', capsize=5, linewidth=1.3)
#     plt.errorbar(errorbar_xvals, evals_avg_69_300['Accuracy'].iloc[errorbar_xvals], yerr=[std_values_dict_300[69][i] for i in errorbar_xvals], fmt='none', ecolor='green', capsize=5, linewidth=1.3)
#     plt.errorbar(errorbar_xvals, evals_avg_71_300['Accuracy'].iloc[errorbar_xvals], yerr=[std_values_dict_300[71][i] for i in errorbar_xvals], fmt='none', ecolor='red', capsize=5, linewidth=1.3)

#     # plt.legend()
#     # plt.text(x=-0.1, y = 0.70, s='Window 200', color='black', bbox=dict(facecolor='white', alpha=0.5))
#     # plt.text(x=-0.1, y = 0.62, s='Window 300', color='black', bbox=dict(facecolor='white', alpha=0.5))
    
#     # Primary legend for subjects
#     subject_legend = plt.legend(loc='upper left')

#     # Custom legend entries for window types
#     from matplotlib.lines import Line2D
#     custom_lines = [Line2D([0], [0], color='black', lw=1.3, linestyle='-'),
#                     Line2D([0], [0], color='black', lw=1.3, linestyle='--')]

#     # Adding the secondary legend
#     window_legend = plt.legend(custom_lines, ['Window 200', 'Window 300'], loc='upper right', bbox_to_anchor=(0.255, 0.81), fontsize='small')
#     plt.gca().add_artist(subject_legend)

#     plt.xlabel('Number of Repetitions used for training')
#     plt.ylabel("Accuracy")
#     plt.xticks(range(5))
#     plt.grid(True)
#     # plt.show()
#     plt.savefig(f'{database}/{TRAIN_TYPE}/Cross_{SPLIT}/TL/{FREEZE}/{filename}_plot_3_amputee.png')
#     plt.close()

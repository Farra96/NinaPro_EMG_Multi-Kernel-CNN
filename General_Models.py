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
    MultiKernelConv2D_grid, TCN_ch_sp_ATT


# INFOS DB3:
# -left hand: [2, 5, 6]
# - SUB 4: -> is over the double from second maximum (worse) DASH SCORE, consider take him down
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
parser.add_argument('-db', '--DATABASE', help='Database to pick', type=str, default='DB2+DB7+DB3')
parser.add_argument('-type', '--TRAIN_TYPE', help = 'Train typology', type=str, default='Standard')
parser.add_argument('-act', '--ACT_FUN', help = 'Activation_Function', type=str, default='relu')

args = parser.parse_args()

# N_SUB = int(args.N_SUB)
TRAIN_TYPE = str(args.TRAIN_TYPE)
DATABASE = str(args.DATABASE)
WND_LEN = int(args.WND_LEN)
NORM_MODE = str(args.NORM_MODE)
NORM_TYPE = str(args.NORM_TYPE)
RECT = args.RECT.lower() == 'true'      # Became false if you pass -rec 'False'
ACT_FUN = str(args.ACT_FUN)



#%% Manual selection
#
# TRAIN_TYPE = 'Reversal'
# DATABASE = 'DB2+DB7+DB3'
# WND_LEN = 300
# NORM_MODE = 'channel'
# NORM_TYPE = 'z_score'
# RECT = True
# ACT_FUN = 'relu'

print(f'TRAIN TYPE: --> {TRAIN_TYPE}\n'
      f'DATABASE :  --> {DATABASE}\n'
      f'WND_LEN  :  --> {WND_LEN}\n'
      f'NORM_MODE:  --> {NORM_MODE}\n'
      f'NORM_TYPE:  --> {NORM_TYPE}\n'
      f'RECTIFY  :  --> {RECT}\n'
      f'ACT_FUN  :  --> {ACT_FUN}')

# %% Database path
database = f'/home/ricardo/{DATABASE}/'
exe_labels = ['Medium wrap', 'Lateral', 'Extensions type', 'Tripod', 'Power sphere', 'Power disk', 'Prismatic pitch',
              'Index Extension', 'Thumb Adduction', 'Prismatic 4 fingers', 'Wave in', 'Wave out', 'Fist', 'Open hand']

device = torch.device('cuda')

filename = f'wnd_{WND_LEN}_{NORM_TYPE}_{NORM_MODE}_rect_{RECT}'
# %% Left-Handed & Corrections
# Left-handed dropped during creation of dataframe
df = pd.read_csv(database + f'Dataframe/dataframe_wnd_{WND_LEN}.csv',
                 dtype={'sub': np.int8, 'label': np.int8, 'rep': np.int8, 'sample_index': np.uint32})
# Making it lighter
n_channels = [cx for cx in df.columns if 'ch' in cx]
df.loc[:, n_channels] = df.loc[:, n_channels].astype(np.float32)

# %% Normalization
df = Norm_each_sub_by_own_param(df, norm_type=NORM_TYPE, mode=NORM_MODE, rectify=RECT)

# INFOS & CORRECTIONS:
# 1) -> SUBJECT 69 CHANNEL 9 AND 10 ARE ALL ZEROS (sub 69 is the sub number 7 of  DB3, amputee and no space for them (0% of forearm).
# Doing Z_score for each subject and channel separately, ends up with nan in the ch9 and ch10 of this subject.
# 2) -> In DB3 (for some reason subject 1 has done less exercise) -> [ 0,  4,  7,  8,  9, 10, 11, 12, 13]
# It doesn't seem a wrong labeling, because the numbers for each gesture are within the normal range, but I decided to drop him.
# He corresponds to subject 63 in the merged DB2+DB7+DB3
df = df.fillna(0)

#%%
if DATABASE == 'DB3':
    df = df.query('sub != 1')
    left_hand_DB3 = [2, 5, 6]
    df.drop(df[df['sub'].isin(left_hand_DB3)].index, inplace=True)
elif DATABASE == 'DB2+DB7+DB3':     # here i don't need to drop left-handed, because i did it a creation of dataframe
    df = df.query('sub != 63')

#TODO: may insert him (and maybe subject 69 aka subject 7 of DB3 with 0 forearm in sub_to_test


n_subject, n_rep = np.unique(df['sub']), np.unique(df['rep'])

# Select sub to test
random.seed(29)
sub_to_test = random.sample(list(n_subject), k= round(len(n_subject)/5))  # for TL framework
# [42, 6, 28, 47, 66, 70, 21, 7, 38, 71, 31, 32]    -> 12 DB2+DB7+DB3
# [6, 28, 21, 7, 38]    -> 5 subject in DB2
# sub_to_test_df = df[df['sub'].isin(sub_to_test)]
# Drop subject to test and to apply TL
df = df[~(df['sub'].isin(sub_to_test))]

n_subject = np.unique(df['sub'])    # Rewrite variable after dropping subject to be tested in TL

# %% Fixing and ordering "sub" column numbers [from 0 to X] for domain classifier.
if TRAIN_TYPE == 'Reversal':
    # Extract unique values and count their occurrences
    unique_values, counts = np.unique(df['sub'], return_counts=True)

    # Create the ordered vector using the counts of unique values
    ordered_sub = np.concatenate([np.full(count, i) for i, count in enumerate(counts)])
    # Rewriting df['sub'] to be from 0 to n_sub (in order to classify the domain with cross-entropy loss)
    df['sub'] = ordered_sub.astype(np.uint8)

    # Reset index
    df = df.reset_index(drop=True)
    df = df.set_index(pd.RangeIndex(start=0, stop=len(df)))
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')

    #%% Create 5 groups for DA
    # The following part can be commented to change experiment: do you want the domain classifier to distinguish between
    # 50 subjects OR 5 groups of 10 subjects each?

    n_subject2 = np.unique(df['sub'])
    df['group'] = np.zeros(len(df)).astype(np.uint8)    # Create new column
    # fill 'sub' column with same value for each 10 subjects to create 5 domain (saved as sub for this purpose)
    for i in range(1, round(len(n_subject2)/10) -1):
        first_index = df[df['sub'] == n_subject2[10 * i - 1]].index[-1]
        last_index = df[df['sub'] == n_subject2[10 * (i + 1) - 1]].index[-1]
        df.loc[first_index + 1:last_index, 'group'] = i
    df.loc[last_index+1 :, 'group'] = 4
    df['sub'] = df['group']     # Substitute column
    df = df.drop('group', axis=1)

    filename = filename + '_5_Groups'

#%% Split dataframe
# Splitting Train, val and teste based on Rep
split_mode = 'rep'
train_val_test_rep = [[2, 4, 6], [1, 5], [3]]       # [[1, 2, 5, 6], [4], [3]]
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
test_group = np.unique(df_test[split_mode])[0]  # Usually only one rep or sub is used for testing, modify it accordingly

if len(df_train) + len(df_val) + len(df_test) != len(df):
    IndexError('Something went wrong when splitting dataframe! Some data are not part of either the train, val and test')

del df


# %% DATASETS
# You must pass to the dataloader a groupby dataframe
if TRAIN_TYPE == 'Standard' or TRAIN_TYPE == 'TCN_SAM_CAM':     # 2 output
    train_set = Pandas_Dataset(df_train.groupby('sample_index'))
    valid_set = Pandas_Dataset(df_val.groupby('sample_index'))

elif TRAIN_TYPE == 'Reversal':          # 3 output
    train_set = Pandas_Dataset(df_train.groupby('sample_index'), return_sub=True)
    valid_set = Pandas_Dataset(df_val.groupby('sample_index'), return_sub=True)

elif TRAIN_TYPE == 'Triplet':           # 4 output
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
num_workers = 8
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
elif ACT_FUN == 'gelu':
    ACT_FUN = nn.GELU()


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
if TRAIN_TYPE == 'Standard' or TRAIN_TYPE == 'Triplet' or TRAIN_TYPE == 'JS':
    model = MKCNN_grid(net)

elif TRAIN_TYPE == 'Reversal':
    model = MKCNN_grid_AN(net, num_domains=len(np.unique(df_train['sub'])))

elif TRAIN_TYPE == 'TCN_SAM_CAM':
    l2_factor = 0.2
    model = TCN_ch_sp_ATT(wnd_len=WND_LEN, n_classes=n_classes, l2_lambda=l2_factor)


model = model.to(device)


# %% Loss Optim and scheduler
# Define Loss functions
cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean').to(device)
# triplet = nn.TripletMarginLoss(reduction='mean', margin=1, p=2)

# Define Optimizer
learning_rate = 0.0002
# changed beta values from (0.5,0.999) to (0.9,0.999)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# # Define Scheduler
precision = 1e-6
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=.2,
                                                       patience=5, verbose=True, eps=precision)

print(f'Model created with Num params: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
# %% Training and validation
num_epochs = 120
if TRAIN_TYPE == 'Standard' or TRAIN_TYPE == 'TCN_SAM_CAM':
    best_weights, tr_losses, val_losses = train_model_standard(model=model, loss_fun=cross_entropy_loss,
                                                               optimizer=optimizer, scheduler=scheduler,
                                                               dataloaders={"train": train_set_generator,
                                                                            "val": valid_set_generator},
                                                               num_epochs=num_epochs, precision=precision,
                                                               patience=10, patience_increase=8)
elif TRAIN_TYPE == 'Triplet':
    best_weights, tr_losses, val_losses = train_model_triplet(model=model, loss_fun=cross_entropy_loss,
                                                              optimizer=optimizer, scheduler=scheduler,
                                                              dataloaders={"train": train_set_generator,
                                                                           "val": valid_set_generator},
                                                              num_epochs=int(num_epochs/2), precision=precision,
                                                              patience=10, patience_increase=8, beta=0.5)

elif TRAIN_TYPE == 'Reversal':
    best_weights, tr_losses, val_losses, tr_dom_losses, val_dom_losses, tr_task_losses, val_task_losses = \
        train_model_reversal_gradient(model=model, loss_fun=cross_entropy_loss, loss_fun_domain=cross_entropy_loss,
                                      lamba=0.5, optimizer=optimizer, scheduler=scheduler,
                                      dataloaders={"train": train_set_generator, "val": valid_set_generator},
                                      num_epochs=num_epochs , precision=precision, patience=10, patience_increase=15)
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
path = database + f'/{TRAIN_TYPE}/Cross_{split_mode}'
if not os.path.exists(path):
    os.makedirs(path)

if TRAIN_TYPE == 'Reversal':
    if net['Output2'] == MultiKernelConv2D_grid:
        filename = filename + '_MK'
    elif net['Output2'] == nn.Flatten:
        filename = filename + '_LL'


# Save state dict of the model
if not os.path.exists(path + f'/Best_States/'):
    os.makedirs(path + f'/Best_States/')
torch.save(best_weights['state_dict'],path + f'/Best_States/state_dict_{filename}.pth')


# %% PlotLoss
if not os.path.exists(path + '/Plot/'):
    os.makedirs(path + '/Plot/')

PlotLoss(tr_losses, val_loss=val_losses,
         title=f'Cross-{split_mode} {test_group}',
         path_to_save=path + '/Plot/',
         filename=f'Cross_{split_mode}_{filename}.png')

# For reversal gradient and JS + center loss
if TRAIN_TYPE == 'Reversal' or TRAIN_TYPE == 'JS':
    PlotLoss(tr_dom_losses, val_loss=val_dom_losses,
             title=f'Cross-{split_mode} on rep {test_group}',
             path_to_save=path + '/Plot/',
             filename=f'Cross_{split_mode}_{filename}_DOMAIN_ONLY.png')

    PlotLoss(tr_task_losses, val_loss=val_task_losses,
             title=f'Cross-{split_mode} on rep {test_group}',
             path_to_save=path + '/Plot/',
             filename=f'Cross_{split_mode}_{filename}_TASK_ONLY.png')

if 'tr_center_losses' in locals():
    # Only for JS + Center Loss
    PlotLoss(tr_center_losses, val_loss=val_center_losses,
             title=f'Cross-{split_mode} on rep {test_group}',
             path_to_save=path + '/Plot/',
             filename=f'Cross_{split_mode}_{filename}_Center_Loss.png')

# %% Conf Matrix
if not os.path.exists(path + '/Conf_Matrix/'):
    os.makedirs(path + '/Conf_Matrix/')

# Evaluation
softmax_block = nn.Softmax(dim=1)
y_true = []
y_pred = []


model.eval()
with torch.no_grad():
    for inputs, labels in test_set_generator:    #!!!!!!!!!!!!!!!! REMEMBER TO CHANGE N_INPUT FOR DATALOADER TYPE
        # inputs = torch.swapaxes(inputs, 2, 1)  # -> convert from [10,20] to [20,10]
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
    plot_confusion_matrix(cm, target_names=exe_labels, title=f'Confusion Matrix for {split_mode} {test_group}',
                          path_to_save=path + f'/Conf_Matrix/{filename}.png')

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
header_net = ['Name', f'Tested_{split_mode}', 'Best Val Loss', 'Accuracy', 'Kappa', 'F1_score', 'Best Epoch',
              'Average Inference Time [ms]', 'Norm_Type', 'Norm_Mode', 'Rect', f'Valid_{split_mode}']

# Open the CSV file and write the headers and row of values
with open(path + '/Evals.csv', 'a', newline='') as myFile:
    writer = csv.writer(myFile)
    if myFile.tell() == 0:
        writer.writerow(header_net)
    # Create the row of values
    row = [filename, test_group, min(val_losses),
           metrics.accuracy_score(y_true=y_true, y_pred=y_pred),
           metrics.cohen_kappa_score(y1=y_true, y2=y_pred, weights='quadratic'),
           metrics.f1_score(y_true=y_true, y_pred=y_pred, average='macro'),
           best_weights['epoch'], avg_time,
           NORM_TYPE, NORM_MODE, str(RECT), valid_group]
    writer.writerow(row)
print(f'Results Saved in -> {path}/Evals.csv')

# Check results
# myFile = open(database + f'Cross_sub/Best_States/Sub_{sub}.csv', 'r')
# print("The content of the csv file is:")
# print(myFile.read())
# myFile.close()


#%%
import sys
import os
import numpy as np
import argparse
import pandas as pd

# Custom Libraries
sys.path.append('/home/ricardo/MKCNN')

from Data_Processing_Utils import Unpack_Database, Build_Sub_Dataframe, mapping_labels, \
    Get_Sub_Norm_params, windowing_Dataframe, PlotLoss, plot_confusion_matrix, plot_power_spectrum_1d, \
    butter_bandpass_filter, iir_notch_filt

#%% INFOS
# - Set variable DATABASE NUMBER
# - Mapping label to be consistent between databases is performed with function "windowing_dataframe"
# - Excluded left handed subjcets, due to missing documentation about data consistency with right ones. 
 
 #%% Argument to be passed

parser = argparse.ArgumentParser(description='Select Database to build and window lenght')

parser.add_argument('-db', '--DATABASE_NUM', help='Database to pick', type=int, default=2)
parser.add_argument('-wnd', '--WND_LEN', help='Windows length', type=int, default=300)

args = parser.parse_args()

DATABASE_NUM = int(args.DATABASE_NUM)
WND_LEN = int(args.WND_LEN)

print(70*'#', 2*'\n', 
      f'Building Dataframe Number {DATABASE_NUM} with window length {WND_LEN}', 
      2*'\n', 70*'#')
# Manual selecton
# DATABASE = 2
# WND_LEN = 300

# %% Unpack data
database = f'/home/riccardo/EMG_data/DB{DATABASE_NUM}/'
# Unpack_Database(database, database_number=DATABASE_NUM)

 #%% Delete FutureWarning
import warnings
warnings.simplefilter(action='ignore', category = FutureWarning) #-> if pandas is bothering you

#%% Build dataframe
# Excluding left-handed Subjects as there are no specifications if they saved the data correctly (channels need to be mapped in the right position). 
# the order of channels of the first 8 circular-shape electrodes is different in the two arms.
if DATABASE_NUM == 2:
    left_hand = [4, 13, 22, 25, 26]   # DB2
elif DATABASE_NUM == 3:
    left_hand = [2, 5, 6]   # DB3
elif DATABASE_NUM == 7:
    left_hand = [1, 4, 5, 14]   # DB7
# Those are the numbers of exercise selected for this project, change them at your wish.
exe_list = {22, 34, 36, 30, 27, 37, 31, 21, 8, 24, 13, 14, 5, 6}  # -> from DB2 -> to MeganePro
exe_labels = ['Medium wrap', 'Lateral', 'Extensions type', 'Tripod', 'Power sphere', 'Power disk', 'Prismatic pich',
              'Index Extension', 'Thumb Adduction', 'Prismatic 4 fingers', 'Wave in', 'Wave out', 'Fist', 'Open hand']


# database + '/Formatted' is where is automatically saved the data from the function: Unpack_Database()

Build_Sub_Dataframe(Unpacked_dataset_path=database + '/Formatted', exe_list_to_keep=exe_list, sub_to_discard=left_hand)

# %% Load database
os.chdir(database + '/Dataframe/')
df = pd.read_csv('dataframe.csv', dtype={'Sub': np.int8, 'Exe_Num': np.int8, 'Rep_Num': np.int8})

n_subject = np.unique(df['Sub'])
n_channels = [a for a in df.columns if 'Ch' in a]
n_exe = np.unique(df['Exe_Num'])

# Power spectrum check (It can be possible to apply an off-set compensation, not explored in this work).
# CHx = df['Ch2']
# plot_power_spectrum_1d(CHx, 2000, freq_range=(5, 160))

#%% Filtering dataframe
# Butterworth
# fs = 2000  # Sampling frequency
# lowcut = 15  # Lower cutoff frequency
# highcut = 500  # Upper cutoff frequency
# order = 4  # Filter order

# # Notch filter for power line (already done for the Ninapros databases).
# # f0 = 50  # Frequency to cut
# # quality = 30  # Quality factor

# for ch in n_channels:
#     df[ch] = butter_bandpass_filter(df[ch], lowcut, highcut, fs, order, plot_sign=True)
    # df[ch] = iir_notch_filt(df[ch], fs, f0, quality, plot_sign=True)

# CHx = df['Ch2']
# plot_power_spectrum_1d(CHx, 2000, freq_range=(5, 160))

#%% Check normal distributions of the channels (to be considered normal:  -2 < sk < 2     &     -7 < kur <7  )
# cols = [col for col in df.columns if 'ch' in col]
#
# sk = np.empty(shape=(len(n_subject), len(n_exe),len(n_channels)))
# kur = sk
# # Subject channel distribution
# c1, c2, c3 = 0, 0, 0
# for sbj in n_subject:
#     for ex in n_exe:
#         df_sbj = df.query('Sub == @sbj and Exe_Num == @ex')
#         for ch in cols:
#             sk[c1, c2, c3] =  scipy.stats.skew(df_sbj[ch])
#             kur[c1, c2, c3] = scipy.stats.kurtosis(df_sbj[ch])
#             c3 = c3 +1
#         c2 = c2 +1
#     c1 = c1 +1

#%% Getting params for real-time considerations and saving them into csv file to facilitate on-line application.
# Consider make it a function
# Sub_Grouped = df.groupby('Sub')
# for norm_type in ['zero_to_one', 'minus_one_to_one', 'z_score']:
#     norm_param_df = None
#     norm_param_xCH = None
#     for sub in n_subject:
#         sub_x = Sub_Grouped.get_group(sub) # -> getting sub_based Dataframe
#         arr_to_norm = sub_x.loc[:, [f'Ch{ch}' for ch in range(1,len(n_channels)+1)]]
#         sub_x_norm_param = Get_Sub_Norm_params(arr_to_norm, norm_type=norm_type, mode='sub')
#         sub_x_norm_param_xCH = Get_Sub_Norm_params(arr_to_norm, norm_type=norm_type, mode='channels')

#         sub_x_norm_param.pop('Norm_Type')
#         # -> I will save this value just one time in the name; consider delete NormType from the function
#         sub_x_norm_param_xCH.pop('Norm_Type')

#         sub_x_norm_param['N_Sub'] = sub
#         sub_x_norm_param_xCH['N_sub'] = sub
#         if norm_param_df is None:
#             norm_param_df = pd.DataFrame.from_dict([sub_x_norm_param])
#             norm_param_xCH = pd.DataFrame.from_dict(sub_x_norm_param_xCH)
#         else:
#             tmp = pd.DataFrame.from_dict([sub_x_norm_param])
#             norm_param_df = pd.concat([norm_param_df,tmp])
#             tmp = pd.DataFrame.from_dict(sub_x_norm_param_xCH)
#             norm_param_xCH = pd.concat([norm_param_xCH, tmp])
#         # Saving a dataframe with params for each channel and each
#         if not os.path.exists(database + 'Sub_Norm_Params/'):
#             os.makedirs(database + 'Sub_Norm_Params/')
#         norm_param_df.to_csv(database + 'Sub_Norm_Params/' + norm_type + '.csv')
#         norm_param_xCH.to_csv(database + 'Sub_Norm_Params/' + norm_type + '_xCH.csv')

#################################################################
# YOU SHOULD NORMALIZE THE DATA BEFORE THE DATALOADER TO SPEED UP THE OFFLINE TRAINING
# Code for normalizing the windows entering the model will be done for the embedded system
################################################################

#%% Windowing signals from dataframe -> Remember to select variable NinaPro_Database_Num to perform correct re-labeling
os.chdir(database)
df = pd.read_csv(database +'/Dataframe/dataframe.csv')
wnd_length, overlap = WND_LEN, 0.75
windowing_Dataframe(df, wnd_length, overlap_perc=overlap, NinaPro_Database_Num=DATABASE_NUM, drop_last=False,
                     path_to_save=database + '/Dataframe/', filename=f'dataframe_wnd_{wnd_length}.csv')

# For training, refers to General_Models

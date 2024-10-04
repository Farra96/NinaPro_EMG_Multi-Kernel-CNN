import scipy
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import itertools

import torch
from matplotlib.ticker import ScalarFormatter
from sklearn.model_selection import KFold
from scipy.signal import butter, filtfilt, freqz


# %% Path
# NinaPro_data_path = '/home/ricardo/DB2/'
# This function will automatically unpack the different database on the servers

# %%
def Unpack_Database(NinaPro_data_path, database_number: int, path_to_save=None):
    """

    :param NinaPro_data_path: Path where zip files have been unzipped
    :param database_number: int to swap between different ways data have been saved before to zip them
    :param path_to_save: Path were to save the formatted data in the shape of "SubXesYrepZ",
                            if None is selected then the path will be: NinaPro_data_path + 'Formatted/'
    :return: Printing "DONE"
    """

    if path_to_save is None:
        path_to_save = NinaPro_data_path + 'Formatted/'
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    folders = sorted(os.listdir(NinaPro_data_path))

    if database_number == 1:  #################################################################################
        os.chdir(NinaPro_data_path)
        mat_files = sorted(os.listdir())
        mat_files = [mat_name for mat_name in mat_files if '.mat' in mat_name]

        for exe_type in mat_files:
            print(exe_type)
            sub = re.findall(r'\d+', exe_type)[0]
            EXE = int(re.findall(r'\d+', exe_type)[2])
            struct = scipy.io.loadmat(exe_type)

            # Correcting the number in the exercises of DB1, bc they are divided by exe_type
            # So if u just run, u would over-write exercises for each exe_type (1,2 or 3)
            exe_num_to_corr = np.unique(struct['restimulus'])

            # Fixing the ex_num to not over-write samples from previous exercise type
            if EXE == 2:
                exe_num = exe_num_to_corr + 12  # -> 12 classes + rest
            elif EXE == 3:
                exe_num = exe_num_to_corr + 29  # (12+17 classes) = 29 + 2 rest = 31
            else:
                exe_num = exe_num_to_corr
            n_gesture = len(exe_num_to_corr)
            repetitions = np.unique(struct['repetition'])
            if len(repetitions) != 11:  # -> 10 Repetitions for DB1
                print('#' * 100 + '\n Number Repetitions not - consistent')

            ############ COMMON PART ############# consider make it a function
            # Here I get the signal, it is possible to get and save other variables from the struct e.g. -> accelerometers, gloves positions
            for i in range(1, n_gesture):  # Rest class skipped
                stimulus_points = np.where(struct['restimulus'] == exe_num_to_corr[i])[0]
                start_idx = stimulus_points[0]
                rep = 1

                # Extracting repetitions
                for j in range(1, len(stimulus_points)):
                    if (stimulus_points[j] - stimulus_points[j - 1]) > 5:
                        matrix = struct['emg'][start_idx: stimulus_points[j - 1]]

                        file = 'Sub' + sub + 'e' + str(exe_num[i]) + 'rep' + str(rep) + '.npy'
                        np.save(path_to_save + file, matrix)

                        start_idx = stimulus_points[j]
                        rep = rep + 1
                    if j == len(stimulus_points) - 1:  # Filling last stimulus
                        matrix = struct['emg'][start_idx: stimulus_points[j]]

                        file = 'Sub' + sub + 'e' + str(exe_num[i]) + 'rep' + str(rep) + '.npy'
                        np.save(path_to_save + file, matrix)


    elif database_number == 2 or database_number == 3:  #################################################################################
        for subjects in folders:
            if 'DB3' in subjects or 'DB2' in subjects:

                os.chdir(NinaPro_data_path + subjects)
                mat_files = sorted(os.listdir())
                mat_files = [mat_name for mat_name in mat_files if ('.mat' in mat_name) and ('E3' not in mat_name)]
                # 'E3' refers to Force Level recognition exercises

                sub = re.findall(r'\d+', subjects)[1]
                for exe_type in mat_files:
                    print(exe_type)
                    struct = scipy.io.loadmat(exe_type)
                    exe_num = np.unique(struct['restimulus'])
                    n_gesture = len(exe_num)
                    repetitions = np.unique(struct['repetition'])
                    if len(repetitions) != 7:  # -> 6 Repetition for DB3 and DB2
                        print('#' * 100 + '\n Number Repetitions not - consistent')

                    ############ COMMON PART ############# consider make it a function
                    # Here I get the signal, it is possible to get and save other variables from the struct e.g. -> accelerometers, gloves positions
                    for i in range(1, n_gesture):  # Rest class deleted
                        stimulus_points = np.where(struct['restimulus'] == exe_num[i])[0]
                        start_idx = stimulus_points[0]
                        rep = 1

                        # Extracting repetitions
                        for j in range(1, len(stimulus_points)):
                            if (stimulus_points[j] - stimulus_points[j - 1]) > 5:
                                matrix = struct['emg'][start_idx: stimulus_points[j - 1]]

                                file = 'Sub' + sub + 'e' + str(exe_num[i]) + 'rep' + str(rep) + '.npy'
                                np.save(path_to_save + file, matrix)

                                start_idx = stimulus_points[j]
                                rep = rep + 1
                            if j == len(stimulus_points) - 1:  # Filling last stimulus
                                matrix = struct['emg'][start_idx: stimulus_points[j]]

                                file = 'Sub' + sub + 'e' + str(exe_num[i]) + 'rep' + str(rep) + '.npy'
                                np.save(path_to_save + file, matrix)


    elif database_number == 7:
        os.chdir(NinaPro_data_path)
        mat_files = [mat_name for mat_name in folders if ('.mat' in mat_name) and ('E3' not in mat_name)]
        for exe_type in mat_files:
            print(exe_type)
            sub = re.findall(r'\d+', exe_type)[0]
            struct = scipy.io.loadmat(exe_type)
            exe_num = np.unique(struct['restimulus'])
            n_gesture = len(exe_num)
            repetitions = np.unique(struct['repetition'])
            if len(repetitions) != 7:  # -> 6 Repetition for DB7
                print('#' * 100 + '\n Number Repetitions not - consistent')

            ############ COMMON PART ############# consider make it a function
            # Here I get the signal, it is possible to get and save other variables from the struct e.g. -> accelerometers, gloves positions
            for i in range(1, n_gesture):  # Rest class deleted
                stimulus_points = np.where(struct['restimulus'] == exe_num[i])[0]
                start_idx = stimulus_points[0]
                rep = 1

                # Extracting repetitions
                for j in range(1, len(stimulus_points)):
                    if (stimulus_points[j] - stimulus_points[j - 1]) > 5:
                        matrix = struct['emg'][start_idx: stimulus_points[j - 1]]

                        file = 'Sub' + sub + 'e' + str(exe_num[i]) + 'rep' + str(rep) + '.npy'
                        np.save(path_to_save + file, matrix)


                        start_idx = stimulus_points[j]
                        rep = rep + 1
                    if j == len(stimulus_points) - 1:  # Filling last stimulus
                        matrix = struct['emg'][start_idx: stimulus_points[j]]

                        file = 'Sub' + sub + 'e' + str(exe_num[i]) + 'rep' + str(rep) + '.npy'
                        np.save(path_to_save + file, matrix)


    elif database_number == 6:  #################################################################################
        for subjects in folders:  # S2_D2_T2 has only two exercise, because all the stimulus points are targeted as 1
            if 'DB6' in subjects:  # So I DELETED S2_D2_T2 due to corruption of data
                os.chdir(NinaPro_data_path + subjects)
                mat_files = sorted(os.listdir())
                mat_files = [mat_name for mat_name in mat_files if '.mat' in mat_name]

                sub = re.findall(r'\d+', subjects)[1]
                for exe_type in mat_files:
                    print(exe_type)
                    day = re.findall(r'\d+', exe_type)[1]
                    daytime = re.findall(r'\d+', exe_type)[2]
                    struct = scipy.io.loadmat(exe_type)
                    exe_num = np.unique(struct['restimulus'])
                    n_gesture = len(exe_num)
                    ############ COMMON PART ############# consider make it a function
                    # Here I get the signal, it is possible to get and save other variables from the struct e.g. -> accelerometers, gloves positions
                    for i in range(1, n_gesture):  # Rest class deleted
                        stimulus_points = np.where(struct['restimulus'] == exe_num[i])[0]
                        start_idx = stimulus_points[0]
                        rep = 1

                        # Extracting repetitions
                        for j in range(1, len(stimulus_points)):
                            if (stimulus_points[j] - stimulus_points[j - 1]) > 5:
                                matrix = struct['emg'][start_idx: stimulus_points[j - 1]]

                                file = 'Sub' + sub + 'e' + str(exe_num[i]) + 'rep' + str(
                                    rep) + 'D' + day + 'T' + daytime + '.npy'
                                np.save(path_to_save + file, matrix)

                                start_idx = stimulus_points[j]
                                rep = rep + 1
                            if j == len(stimulus_points) - 1:  # Filling last stimulus
                                matrix = struct['emg'][start_idx: stimulus_points[j]]

                                file = 'Sub' + sub + 'e' + str(exe_num[i]) + 'rep' + str(
                                    rep) + 'D' + day + 'T' + daytime + '.npy'
                                np.save(path_to_save + file, matrix)

    return print('DONE')


# %% Build a different array for each subjects
def Build_Sub_Dataframe(Unpacked_dataset_path, exe_list_to_keep=np.arange(54), sub_to_discard=None, path_to_save=None):
    """

    :param Unpacked_dataset_path: Data are in the form SubXeYrepZ.npy where it is stored the EMG signals of each channel
    :param exe_list_to_keep:      List of exercise to keep for Hand Gesture Classificaton
    :param sub_to_discard:        Choose if there are subjects to not consider (e.g. left handed, amputee, corrupted data)
    :param path_to_save:          Path to save dataframe files, if none is passed, will be subfolder "/Dataframe"
                                    of Unpacked_dataset_path
    :return:                      Saved Dataframe in the form "Channels, Sub, Exe, Rep"
    """
    if path_to_save is None:
        path_to_save = os.path.join(os.path.dirname(Unpacked_dataset_path)) + '/Dataframe/'
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    os.chdir(Unpacked_dataset_path)
    EMG_data = sorted(os.listdir())
    EMG_data = [mat_name for mat_name in EMG_data if 'Sub' and '.npy' in mat_name]  # 14040 for DB1
    n_channels = min(np.load(EMG_data[5]).shape)

    # Concatenate Numpy array
    SUB, SIGNAL, EXE_NUM, REP = np.array([]), np.empty(shape=(0, n_channels)), np.array([]), np.array([])
    counter = 1
    for name in EMG_data:
        print(f'EMG_data {counter}/ {len(EMG_data)}')
        counter = counter + 1
        sub = np.int8(re.findall(r'\d+', name)[0])
        if sub_to_discard is not None:
            if sub in sub_to_discard:  # Selecting subjects to keep
                continue
        exe_num = np.int8(re.findall(r'\d+', name)[1])
        if exe_list_to_keep is not None:
            if exe_num in exe_list_to_keep:  # Selecting exercise to keep

                EMG = np.load(name).astype(np.float32)
                rep = np.int8(re.findall(r'\d+', name)[2])

                SUB = np.concatenate((SUB, np.full(shape=len(EMG), fill_value=sub)))
                SIGNAL = np.concatenate((SIGNAL, EMG))  # Those parenthesis is needed
                EXE_NUM = np.concatenate((EXE_NUM, np.full(shape=len(EMG), fill_value=exe_num)))
                REP = np.concatenate((REP, np.full(shape=len(EMG), fill_value=rep)))

    # Creating dataframe
    columns = [f'Ch{ch}' for ch in range(1, n_channels + 1)]
    df = pd.DataFrame(data=SIGNAL, columns=columns, dtype=np.float32)
    df['Sub'] = SUB
    df['Exe_Num'] = EXE_NUM
    df['Rep_Num'] = REP

    df.to_csv(path_to_save + 'dataframe.csv', columns=df.columns)
    print('DONE')


# %% Save sub norm param to pick data from for real time applications
def Get_Sub_Norm_params(arr_2D, norm_type, mode=None):
    """

    :param arr_2D:    Array (time x channel) to get params from
    :param norm_type: Type of Normalization to get param from
    :param mode:      If None, Get params from array
                      if 'Channels' -> normalize each channel separately
    :return INFO:     Dictionary with the required information
    """

    if (mode is None) or (mode == 'sub'):
        if norm_type == 'zero_to_one' or norm_type == 'minus_one_to_one':
            MAX, MIN = np.max(arr_2D), np.min(arr_2D)
            # -> 2 times mx().max() bc if u pass pandas dataframe is going to do it for Columns
            INFO = {'Norm_Type': norm_type, 'Max': MAX, 'Min': MIN}
        elif norm_type == 'z_score':
            MEAN, STD = np.mean(arr_2D), np.std(arr_2D)
            INFO = {'Norm_Type': norm_type, 'Mean': MEAN, 'STD': STD}
        else:
            raise ValueError('Normalization type not supported!')

    elif mode == 'channels':
        if arr_2D.shape[0] < arr_2D.shape[1]:
            arr_2D = arr_2D.T
            print('*' * 100 + '\nArray transposed!\n' + '*' * 100)
        # ATTENTION:-> Sometimes some channels are zeros (no space for ch_X in some amputee or reference ch in devices)
        if (norm_type == 'zero_to_one') or (norm_type == 'minus_one_to_one'):
            MAX, MIN = arr_2D.max(axis=0), arr_2D.min(axis=0)  # -> 1 value for each channel
            INFO = {'Norm_Type': norm_type, 'Max_chs': MAX, 'Min_chs': MIN}
        elif norm_type == 'z_score':
            MEAN, STD = arr_2D.mean(axis=0), arr_2D.std(axis=0)
            INFO = {'Norm_Type': norm_type, 'Mean_ch': MEAN, 'STD_ch': STD}
        else:
            raise ValueError('Normalization type not supported!')

    else:
        raise ValueError('Mode not found! Choose between: "sub" or "channels", default is "sub"')

    return INFO


# %% Normalize 2D array
def NormalizeData(arr_2D, norm_type=None, mu = 2.048):  # mu -> https://ieeexplore.ieee.org/document/9422807
    """
    :param arr_2D:      Array (time x channel) or Dataframe Object to normalize
    :param norm_type:   Type of Normalization
    :param mu:          μ parameter for μ-law normalization (default=255, here 2,048)
    :return:            Normalized 2D_signal
    """
    dataframe = False

    if isinstance(arr_2D, pd.DataFrame):
        dataframe = True
        cols = arr_2D.columns
        index = arr_2D.index
        arr_2D = arr_2D.values

    if norm_type == 'zero_to_one':
        denominator = (np.max(arr_2D) - np.min(arr_2D))
        if denominator == 0:
            denominator = 1
            print('Max & Min of array to norm are the same')
        arr_2D = (arr_2D - np.min(arr_2D)) / denominator
    elif norm_type == 'minus_one_to_one':
        denominator = (np.max(arr_2D) - np.min(arr_2D))
        if denominator == 0:
            denominator = 1
            print('Max & Min of array to norm are the same')
        arr_2D = 2 * ((arr_2D - np.min(arr_2D)) / denominator) - 1
    elif norm_type == 'z_score':
        arr_2D = scipy.stats.zscore(arr_2D, nan_policy='omit')
    elif norm_type == 'mu_law':
        arr_2D = np.sign(arr_2D) * np.log1p(mu * np.abs(arr_2D)) / np.log1p(mu)
    else:
        print('Normalization type not found!')

    if dataframe:
        arr_2D = pd.DataFrame(data=arr_2D, columns=cols, index=index)
        return arr_2D
    else:
        return arr_2D


# %% Normalize dataframe column ch
def Norm_each_sub_by_own_param(df, mode='sub', norm_type = None,
                               rectify=False):  # , path_to_pick_sub_norm_param = None):
    '''

    :param df:          Data to normalize, need a column named 'sub'
    :param mode:        Choose between 'sub' or 'channel' to choose if normalizing each channel separately
    :param norm_type:   Choose between 'minus_one_to_one', 'zero_to_one' or 'z_score'
    :param rectify:     Boolean, if yes it will be taken the absolute value of the signal before normalizing
    :return:            The dataframe with the normalized data
    '''

    # -> this is for inference in the train_model_standard
    # norm_params = pd.read_csv(path_to_pick_sub_norm_param + norm_type + '.csv')

    data = df.copy()
    n_sub = np.unique(df['sub'])
    sub_grouped = df.groupby('sub')
    chs = [col for col in df.columns if 'ch' in col]

    if rectify:
        print('RECTIFICATION OF THE DATAFRAME')
        data[chs] = data[chs].abs()
        print('DONE')

    if mode == 'sub':
        for i in n_sub:
            print(f'Norm SUB: {i}')
            sub_X = sub_grouped.get_group(i)
            arr_to_norm = sub_X.filter(like='ch', axis=1)
            normed_arr = NormalizeData(arr_to_norm, norm_type=norm_type)
            data.loc[normed_arr.index, chs] = normed_arr
    elif mode == 'channel':
        for i in n_sub:
            print(f'\nNorm SUB: {i}')
            sub_X = sub_grouped.get_group(i)
            arr_to_norm = sub_X.filter(like='ch', axis=1)
            normed_arr = arr_to_norm.copy()
            for j in arr_to_norm.columns:
                print('Norm CH:', j)
                normed_arr[j] = NormalizeData(normed_arr[j], norm_type=norm_type)
                data.loc[normed_arr.index, j] = normed_arr
    else:
        print('Normalization mode not found!')

    print('DONE')

    return data

def Norm_each_sub_label_by_own_param(df, mode='sub', norm_type=None, rectify=False):
    '''
    Normalizes each channel based on each subject and each label.
    
    :param df:          Data to normalize, must contain columns named 'sub' and 'label'
    :param mode:        Choose between 'sub' or 'channel' to determine if normalizing each channel separately
    :param norm_type:   Choose between 'minus_one_to_one', 'zero_to_one', or 'z_score'
    :param rectify:     Boolean, if True, takes the absolute value of the signal before normalizing
    :return:            The dataframe with the normalized data
    '''

    data = df.copy()
    n_sub = np.unique(df['sub'])
    chs = [col for col in df.columns if 'ch' in col]

    if rectify:
        print('RECTIFICATION OF THE DATAFRAME')
        data[chs] = data[chs].abs()
        print('DONE')

    # Group by both subject and label
    sub_label_grouped = df.groupby(['sub', 'label'])

    if mode == 'sub':
        for (sub, label), group in sub_label_grouped:
            print(f'Norm SUB: {sub}, LABEL: {label}')
            arr_to_norm = group.filter(like='ch', axis=1)
            normed_arr = NormalizeData(arr_to_norm, norm_type=norm_type)
            data.loc[group.index, chs] = normed_arr
            
    elif mode == 'channel':
        for (sub, label), group in sub_label_grouped:
            print(f'\nNorm SUB: {sub}, LABEL: {label}')
            arr_to_norm = group.filter(like='ch', axis=1)
            normed_arr = arr_to_norm.copy()
            for j in arr_to_norm.columns:
                print(f'Norm CH: {j}')
                normed_arr[j] = NormalizeData(normed_arr[j], norm_type=norm_type)
                data.loc[group.index, j] = normed_arr
                
    else:
        print('Normalization mode not found!')

    print('DONE')
    return data
# %% Database 1,3,6,7

def mapping_labels(lab, NinaPro_Database_Num: int):
    """
    Maps old labels to new labels based on the specified NinaPro Database Number.

    Parameters:
    - lab (int): Original label to be mapped.
    - NinaPro_Database_Num (int): Identifier for the NinaPro Database.

    Returns:
    - int or None: Mapped label, or None if the original label is not found.

    Raises:
    - ValueError: If the NinaPro Database Number is not supported.
    """
    label_maps = {
        1: {34: 0, 46: 1, 48: 2, 42: 3, 39: 4, 49: 5, 43: 6, 33: 7, 9: 8, 36: 9, 25: 10, 26: 11, 17: 12, 18: 13},
        2: {22: 0, 34: 1, 36: 2, 30: 3, 27: 4, 37: 5, 31: 6, 21: 7, 8: 8, 24: 9, 13: 10, 14: 11, 5: 12, 6: 13},
        3: {22: 0, 34: 1, 36: 2, 30: 3, 27: 4, 37: 5, 31: 6, 21: 7, 8: 8, 24: 9, 13: 10, 14: 11, 5: 12, 6: 13},
        7: {22: 0, 34: 1, 36: 2, 30: 3, 27: 4, 37: 5, 31: 6, 21: 7, 8: 8, 24: 9, 13: 10, 14: 11, 5: 12, 6: 13},
        6: {1: 0, 3: 1, 4: 2, 6: 3, 9: 4, 10: 5, 11: 6}
    }

    try:
        mapped_lab = label_maps[NinaPro_Database_Num].get(lab)
        if mapped_lab is None:
            print(f'Mapping Failed! Class {lab} not found for Database {NinaPro_Database_Num}.')
        return mapped_lab
    except KeyError:
        
        raise ValueError(f'NinaPro Database {NinaPro_Database_Num} NOT FOUND! No re-labeling performed.')

# %%  Windowing signal
def windowing_signal(arr_2d, wnd_length: int, overlap_perc, drop_last=True):
    """

    :param arr_2d: Array shape Signal x Channels to be windowed
    :param wnd_length: length of the window
    :param overlap_perc: amount of overlap in percentage
    :param drop_last: if true drop last samples, if false drop first & last samples in equal amount

    :return: Return windowed array in 3-D (n_wnd, signal x channel)
    """

    if arr_2d.shape[1] > arr_2d.shape[0]:
        arr_2d = np.transpose(arr_2d)

    wnd_step = wnd_length - wnd_length * overlap_perc
    n_window = int((arr_2d.shape[0] - wnd_length) / wnd_step)
    data_left = int((arr_2d.shape[0] - wnd_length) % wnd_step)

    if drop_last:
        arr_2d = arr_2d[0: len(arr_2d) - data_left]
    else:
        if data_left % 2 == 0:
            arr_2d = arr_2d[int(data_left / 2): int(
                len(arr_2d) - data_left / 2)]  # Added len() bc if data_left = 0 -> arr_2d = []
        else:
            arr_2d = arr_2d[int(data_left / 2) + 1: int(len(arr_2d) - int(data_left / 2))]

    windowed = np.zeros(shape=(n_window, wnd_length, arr_2d.shape[1]))

    for wnd in range(n_window):
        windowed[wnd] = arr_2d[int(wnd * wnd_step): int(wnd * wnd_step) + wnd_length]

    return windowed


# %% Windowing Dataframe
def windowing_Dataframe(df, wnd_length: int, overlap_perc, NinaPro_Database_Num= None, drop_last=True, path_to_save=None,
                        filename=None):
    """

    :param df:                      Pandas df to be windowed
    :param wnd_length:              Length of thw window
    :param overlap_perc:            Overlapping percentage of the windows
    :param NinaPro_Database_Num:    Variable to perform a relabeling of the pool of subclasses chosen
    :param drop_last:               If True, it drops the last window if it has no enough point to reach window_len
    :param path_to_save:            Path to save the dataframe once windowed
    :param filename:                Filename of the saved dataframe
    :return:
    """
    if path_to_save is None:
        path_to_save = f'/home/ricardo/DB{NinaPro_Database_Num}/Dataframe/'
    if filename is None:
        filename = f'dataframe_wnd_{wnd_length}.csv'
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    n_sub = np.unique(df['Sub'])
    n_channels = [ch for ch in df.columns if 'Ch' in ch]
    n_exe_default = np.unique(df['Exe_Num'])
    n_rep_default = np.unique(df['Rep_Num'])

    signal_index = df.set_index(['Sub', 'Exe_Num', 'Rep_Num'])
    signal_index.sort_index(inplace=True)

    windows_arr = np.empty(shape=(0, len(n_channels))).astype(np.float32)
    sub_arr, label_arr, rep_arr = np.array([]), np.array([]), np.array([])
    n_samples = int(0)

    for sub in n_sub:
        print(f'Windowing subject: {sub}')
        n_exe = np.unique(df.loc[df['Sub'] == sub]['Exe_Num'])  # Some subject may have done fewer exercise type
        if not np.array_equiv(n_exe_default, n_exe):
            print(f'SUB: {int(sub)} has done fewer exercise: {len(n_exe)} / {len(n_exe_default)}')
        for exe in n_exe:

            n_rep = np.unique(df.loc[df['Sub'] == sub]['Rep_Num'])
            if not np.array_equiv(n_rep_default, n_rep):  # Some subject may have done fewer repetitions
                print(f'SUB: {int(sub)} has done fewer repetition: {len(n_rep)} / {len(n_rep_default)}')

            # -> Correcting label based on database and chosen classes to use for classification
            label = mapping_labels(exe, NinaPro_Database_Num=NinaPro_Database_Num)

            for rep in n_rep:
                arr_to_wnd = signal_index.loc[sub, exe, rep]  # -> Extracting SubXesYrepZ
                windowed = windowing_signal(arr_to_wnd, wnd_length, overlap_perc, drop_last)  # -> Windowing 2d array
                wnd_unpacked = windowed.reshape(-1, windowed.shape[-1])  # -> Flattening 3D (n_wnd x wnd_len x ch) to 2D

                n_samples = n_samples + windowed.shape[0]
                # -> Concatenating
                windows_arr = np.concatenate((windows_arr, wnd_unpacked)).astype(np.float32)
                label_arr = np.concatenate((label_arr, np.full(shape=wnd_unpacked.shape[0],
                                                               fill_value=label))).astype(np.int8)
                sub_arr = np.concatenate((sub_arr, np.full(shape=wnd_unpacked.shape[0],
                                                           fill_value=sub))).astype(np.int8)
                rep_arr = np.concatenate((rep_arr, np.full(shape=wnd_unpacked.shape[0],
                                                           fill_value=rep))).astype(np.int8)

    # Building DataFrame
    columns = [f'ch{ch}' for ch in range(1, len(n_channels) + 1)]
    wnd_df = pd.DataFrame(data=windows_arr, columns=columns)
    wnd_df['sub'], wnd_df['label'], wnd_df['rep'] = sub_arr, label_arr, rep_arr
    # Adding Indexes for each windows
    samples_index_arr = np.empty(len(wnd_df)).astype(np.uint32)
    for idx in range(n_samples):
        samples_index_arr[idx * wnd_length: (idx + 1) * wnd_length] = idx
    wnd_df['sample_index'] = samples_index_arr
    # Could add repetitions if needed

    wnd_df = wnd_df.set_index('sample_index')

    wnd_df.to_csv(path_to_save + filename)
    return print(f'Saved in -> {path_to_save}  as -> {filename}')


# %% Train, val & test split function for dataframe
def train_val_test_split_df(dataframe, percentages=None, mode=None, manual_sel=None, seed = None):
    '''

    :param dataframe:     Dataframe to be split by
    :param percentages:   Ordered percentage to divide the dataframe in train, val, test
    :param mode:          Which way to split dataframe, any column value can be used ('sub', 'rep', 'day')
    :param manual_sel:    If not none, select which "mode" is going to be the train val and test.
                          Need to be a list of 3 list, for train val and test respectively

    :return:              train, validation and test dataframe with the same structure of the original one
    '''
    if seed:
        np.random.seed(seed)
    if percentages is None:
        percentages = [0.6, 0.2, 0.2]
    elif sum(percentages) != 1:
        print('Percentages to divide dataframe are not equal to one! you are leaving out data!')

    if percentages[2] == 0:  # Handle the case where test percentage is set to zero
        print("Test percentage is set to zero! \n \
        For code semplicity  and data integraty the validation dataframe will be saved in the third variable returned")
        percentages[2] = percentages [1]
        percentages[1] = 0

    if mode is None:  # random sample
        idx_list = np.unique(dataframe.index)
        n_samples = len(idx_list)
        # shuffle indexes
        np.random.shuffle(idx_list)
        # train val and test sizes
        tr_size, val_size = round(n_samples * percentages[0]), round(n_samples * percentages[1])
        # train, val, and test indexes
        tr_idx, val_idx, test_idx = idx_list[: tr_size], idx_list[tr_size: tr_size + val_size], idx_list[
                                                                                                tr_size + val_size:]
        # train val and test dataframes
        tr_df, val_df, test_df = dataframe.iloc[tr_idx], dataframe.iloc[val_idx], dataframe.iloc[test_idx]

    else:
        if mode not in dataframe.columns:
            raise ValueError(f'Mode {mode} not present in dataframes columns!')
        if manual_sel is not None:
            if len(manual_sel) != 3:
                raise ValueError('Provide a list with 3 ordered list for manual splitting\n'
                                 'Example: manual_sel = [[2,4,6],[1,3],[4]]')
            tr_df = dataframe[dataframe[mode].isin(manual_sel[0])]
            val_df = dataframe[dataframe[mode].isin(manual_sel[1])]
            test_df = dataframe[dataframe[mode].isin(manual_sel[2])]

        else:
            mode_list = dataframe[mode].unique()
            np.random.shuffle(mode_list)
            # train val and test sizes
            tr_size, val_size = round(len(mode_list) * percentages[0]), round(len(mode_list) * percentages[1])
            # train, val, and test indexes
            tr_list, val_list, test_list = mode_list[:tr_size], mode_list[tr_size:tr_size + val_size], mode_list[
                                                                                                       tr_size + val_size:]
            # train val and test dataframes
            tr_df, val_df = dataframe[dataframe[mode].isin(tr_list)], dataframe[dataframe[mode].isin(val_list)]
            test_df = dataframe[dataframe[mode].isin(test_list)]

    if len(tr_df) + len(val_df) + len(test_df) != len(dataframe):
        IndexError(
            'Something went wrong when splitting dataframe! Some data are not part of either the train, val and test')

    return tr_df, val_df, test_df



# %% K-Fold
def kfold_cross_validation(df, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True)
    unique_subs = df['sub'].unique()
    fold_dfs = []

    for train_index, test_index in kf.split(unique_subs):
        train_subs = unique_subs[train_index]
        test_subs = unique_subs[test_index]
        train_df = df[df['sub'].isin(train_subs)].copy()
        test_df = df[df['sub'].isin(test_subs)].copy()


        fold_dfs.append((train_df, test_df))

    return fold_dfs

# %% PlotLoss
def PlotLoss(tr_loss, title=None, val_loss=None, path_to_save=None, filename=None):
    x = np.arange(1, len(tr_loss) + 1)

    # plt.figure(figsize=(160,90), dpi=100)

    plt.plot(x, tr_loss, color='blue')
    plt.yscale('log')
    # plt.ticklabel_format(style='plain', axis='y')
    # plt.ylim([min(tr_loss), y_lim])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()

    if title:
        plt.title(title)

    if val_loss is not None:
        plt.plot(x, val_loss, color='orange')
        plt.legend(['Train', 'Validation'])

    # Format the y-axis tick labels to display numbers instead of exponential notation
    plt.tight_layout()
    # fig = plt.gcf() # -> get current figure
    if path_to_save:
        plt.savefig(os.path.join(path_to_save, filename))
        plt.close()
        return print(f'Plot saved in --> {path_to_save + filename}')
    else:
        plt.show()


# %% Confusion matrix
def plot_confusion_matrix(cm, normalize=True, cmap=None, target_names=None, title=None, path_to_save=None,
                          fig_size=(90, 80), dpi=600):
    plt.Figure(figsize=fig_size, dpi=dpi)

    tot_acc = np.sum(np.diag(cm)) / np.sum(cm)

    if cmap is None:
        cmap = plt.get_cmap('Blues')
    if normalize:
        # cm = np.diag(cm) / np.sum(cm, axis=1)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
    else:
        tick_marks = np.arange(cm.shape[0])

    font_size = round(fig_size[0] / len(tick_marks))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.xticks(tick_marks, np.arange(cm.shape[0]) + 1, fontsize=font_size)  # rotation=45,
    plt.yticks(tick_marks, target_names, fontsize=font_size)
    # ax.set_xticks(tick_marks)
    # ax.set_xticklabels(target_names, rotation=45, fontsize=font_size, ha='center')
    # ax.set_yticks(tick_marks)
    # ax.set_xticklabels(target_names, fontsize=font_size)
    plt.title(title, fontsize=font_size * 1.5)
    # plt.colorbar()

    cb = plt.colorbar()
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(font_size)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.3f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=font_size)
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=font_size)

    plt.tight_layout()
    plt.ylabel('True Label', fontsize=font_size)
    plt.xlabel('Model predictions: accuracy={:0.4f}'.format(tot_acc), fontsize=font_size)
    # , fontsize= font_size * 2) # \n accuracy={:0.4f}'.format(tot_acc),fontsize=font_size)

    if path_to_save:
        os.chdir(os.path.dirname(path_to_save))
        plt.savefig(path_to_save, dpi=dpi)
        plt.close()
        return print(f'Conf_Matrix saved in -->  {path_to_save}')

    else:
        plt.show()
        return print('Image showed')


#%% Radial basis function
def elongated_ellipsoid(distances, sigma_x, sigma_y):
    return np.exp(-((distances[0] / sigma_x)**2 + (distances[1] / sigma_y)**2))
def radial_basis_function(distances, sigma):
    return np.exp(-(distances**2) / (2 * sigma**2))

def radial_mapping8(n_pix, sigma, wnd, plot_circle=False, plot_data=False):
    '''

    :param n_pix: Number of pixel of the square images extracted from circular device
    :param sigma: Parameter that controls the influenze (distance) of each electrode over the others (gaussian centered
    in channels)
    :param wnd: Window (channel X time) that will be converted in squared-Image X time), it needs to be a dataframe
    :param plot_circle: If True, it plots the position and numbers of the electrodes in the image
    :param plot_data:   If True, it plots the data after extrapolating the image based on gaussian weights centered and
    summed over the electrodes
    :return: The data organized ina 3d Matrix where the first 2 dimensions are the n_pix X n_pix image and the third is
    the temporal dimension
    '''

    time_points, num_ch = wnd.shape
    interpolated_data = np.zeros((n_pix, n_pix, time_points))
    x = np.linspace(0, n_pix - 1, n_pix)
    y = np.linspace(0, n_pix - 1, n_pix)
    X, Y = np.meshgrid(x, y)

    # Calculate channel positions on the circumference of the circle
    angles = np.linspace(0, 2 * np.pi, num_ch, endpoint=False)
    channel_radius = n_pix/2 -round(n_pix*0.05)
    channel_x = n_pix / 2 + channel_radius * np.cos(angles)
    channel_y = n_pix / 2 + channel_radius * np.sin(angles)
    # Calculate the distances from each pixel to the channels
    distances_matrix = np.sqrt((X - channel_x[:, np.newaxis, np.newaxis]) ** 2 +
                               (Y - channel_y[:, np.newaxis, np.newaxis]) ** 2)


    weights_matrix = np.zeros((num_ch, n_pix, n_pix))
    for channel in range(num_ch):
        weights_matrix[channel] = radial_basis_function(distances_matrix[channel], sigma)

    for t in range(time_points):
        for i in range(n_pix):
            for j in range(n_pix):
                weights = weights_matrix[:, i, j]
                interpolated_data[i, j, t] = np.sum(wnd.loc[wnd.index[t], :] * weights)

    # Plot the interpolated data at a fixed time point
    if plot_data:
        fig, ax = plt.subplots(figsize=(n_pix, n_pix))
        for t in range(0, time_points, round(time_points/20)):
           ax.imshow(interpolated_data[:, :, t], cmap='jet', origin='lower')
           plt.show()
           plt.colorbar()
           plt.title(f"Interpolated EMG data at time point {t}")
           plt.xlabel("X")
           plt.ylabel("Y")
           plt.pause(0.3)
           plt.draw()

    # Plot circle in the image
    if plot_circle:
        fig, ax = plt.subplots(figsize=(n_pix, n_pix))
        ax.imshow(np.zeros((n_pix, n_pix)), cmap='gray', origin='lower')

        for ch, x, y in zip(range(1, num_ch + 1), channel_x, channel_y):
            circle = plt.Circle((x, y), radius=round(n_pix / 15), facecolor='red')
            ax.add_patch(circle)
            ax.text(x, y, str(ch), color='white', fontsize=2 * n_pix, ha='center', va='center')
        plt.title("Channel Positions")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xlim(0, n_pix - 1)
        plt.ylim(0, n_pix - 1)
        plt.show()

    return interpolated_data

# Visualize samples
def visualize_samples(df, ex_num, chX, n_plot=4):
    df.set_index('sample_index')
    wnd_len = len(df[df['sample_index'] == 0])
    subs = np.random.choice(np.unique(df['sub']), size=2, replace=False)

    sample_index_list1 = np.unique(df[(df['sub'] == subs[0]) & (df['label'] == ex_num)]['sample_index'])
    sample_index_list2 = np.unique(df[(df['sub'] == subs[1]) & (df['label'] == ex_num)]['sample_index'])

    random_samples1 = df[df['sample_index'].isin(np.random.choice(sample_index_list1, size=2, replace=False))]
    random_samples2 = df[df['sample_index'].isin(np.random.choice(sample_index_list2, size=2, replace=False))]

    fig, axs = plt.subplots(int(n_plot / 2), int(n_plot / 2))
    fig.suptitle(f'Exe num {ex_num} in Channel {chX}')
    axs[0, 0].plot(np.arange(wnd_len), random_samples1[chX][0:wnd_len])
    axs[0, 0].set_title(f'Sub {subs[0]}')
    axs[0, 1].plot(np.arange(wnd_len), moving_average(random_samples1[chX][wnd_len:], int(wnd_len / 20)))
    axs[0, 1].set_title(f'Sub {subs[0]}')
    axs[1, 0].plot(np.arange(wnd_len), random_samples2[chX][0:wnd_len])
    axs[1, 0].set_title(f'Sub {subs[1]}')
    axs[1, 1].plot(np.arange(wnd_len), moving_average(random_samples2[chX][wnd_len:], int(wnd_len / 20)))
    axs[1, 1].set_title(f'Sub {subs[1]}')

    return plt.show()

# %% Filtering
def plot_power_spectrum_1d(signal, sampling_rate, freq_range=None):
    spectrum = np.fft.fft(signal)
    power_spec = np.abs(spectrum) ** 2
    frequencies = np.fft.fftfreq(len(signal), 1 / sampling_rate)

    # Only consider the positive half of frequencies
    positive_indices = frequencies >= 0
    frequencies = frequencies[positive_indices]
    power_spec = power_spec[positive_indices]

    if freq_range is not None:
        start_freq, end_freq = freq_range
        freq_indices = np.logical_and(frequencies >= start_freq, frequencies <= end_freq)
        frequencies = frequencies[freq_indices]
        power_spec= power_spec[freq_indices]

    plt.plot(frequencies, power_spec)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('Power Spectrum')
    plt.grid(True)
    plt.show()

def moving_average(a, n=5):
    a = np.array(a)
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret[0:n] = a[0:n]
    return ret / n


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, plot=False, plot_sign=False):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)  # -> ZeroPhase shift Filtering
    if plot:
        freq, h = freqz(b, a, fs=fs)
        # Plot
        fig, ax = plt.subplots(2, 1, figsize=(8, 6))
        ax[0].plot(freq, 20 * np.log10(abs(h)), color='blue')
        ax[0].set_title("Frequency Response")
        ax[0].set_ylabel("Amplitude (dB)", color='blue')
        ax[0].set_xlim([0, 100])
        ax[0].set_ylim([-25, 10])
        ax[0].grid(True)
        ax[1].plot(freq, np.unwrap(np.angle(h)) * 180 / np.pi, color='green')
        ax[1].set_ylabel("Angle (degrees)", color='green')
        ax[1].set_xlabel("Frequency (Hz)")
        ax[1].set_xlim([0, 1000])
        ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
        ax[1].set_ylim([-90, 90])
        ax[1].grid(True)
        plt.show()
    if plot_sign:
        x_axis = [p for p in range(0, len(data), 2)]
        plt.plot(x_axis, data[x_axis], label='Original')
        plt.plot(x_axis, y[x_axis], color='orange', label='Filtered (Butterworth)', alpha=0.5)
        plt.title('Signal vs Filtered one')
        plt.ylabel('Volt (V)')
        plt.xlabel('Time (ms)')
        plt.grid(True)
        plt.show(dpi=10000)
        plt.legend()
    return y


# %% IIR NOCTH
def iir_notch_filt(signal_1D, F_samp, f0, quality, plot_filt=False, plot_sign=False):
    """

    :param signal_1D:
    :param F_samp:    sampling Frequency
    :param f0:        frequency to cut
    :param quality:   quality factor of the filter
    :param plot_filt: print a plot of the filter
    :param plot_sign: print signal vs filtered signal

    :return: signal_filt:   Filtered signal
    """

    b, a = scipy.signal.iirnotch(f0, quality, F_samp)
    signal_filt = scipy.signal.filtfilt(b, a, signal_1D)

    if plot_filt:
        freq, h = scipy.signal.freqz(b, a, fs=F_samp)
        # Plot
        fig, ax = plt.subplots(2, 1, figsize=(8, 6))
        ax[0].plot(freq, 20 * np.log10(abs(h)), color='blue')
        ax[0].set_title("Frequency Response")
        ax[0].set_ylabel("Amplitude (dB)", color='blue')
        ax[0].set_xlim([0, 100])
        ax[0].set_ylim([-25, 10])
        ax[0].grid(True)
        ax[1].plot(freq, np.unwrap(np.angle(h)) * 180 / np.pi, color='green')
        ax[1].set_ylabel("Angle (degrees)", color='green')
        ax[1].set_xlabel("Frequency (Hz)")
        ax[1].set_xlim([0, 100])
        ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
        ax[1].set_ylim([-90, 90])
        ax[1].grid(True)
        plt.show()
    if plot_sign:
        x_axis = [p for p in range(0, len(signal_1D), 2)]
        plt.plot(x_axis, signal_1D[x_axis], label='Original')
        plt.plot(x_axis, signal_filt[x_axis], label='Filtered (IIR notch)', color='orange', alpha=0.7)
        plt.title('Signal vs Filtered one')
        plt.ylabel('Volt (V)')
        plt.xlabel('Time (ms)')
        plt.legend()
        plt.grid(True)
        plt.show(dpi=10000)

    return signal_filt

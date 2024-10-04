import re
import time
import os
import sys
import random
import copy
import csv
import itertools
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from scipy.stats import loguniform
from sklearn import metrics

from Data_Processing_Utils import PlotLoss, plot_confusion_matrix, train_val_test_split_df
from DATALOADERS import Pandas_Dataset, adjust_dataloader


# %% Personalized conv2D with circular padding along channels axis and zero padding along time axis

class Conv2d_Circular8(nn.Module):
    # Due to problem in the built-in function of Conv2d, padding_mode = 'circular', I here made my own
    def __init__(self, in_channels, out_channels, kernel_size, pad_LRTB, stride=1, bias=True, device=None):
        super(Conv2d_Circular8, self).__init__()

        self.conv_no_pad = nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias, stride=stride, device=device)
        self.pad = pad_LRTB

    def forward(self, x):
        # print(x[0][0])
        circ_ch = x[:, :, :, 0:8]
        other_ch = x[:, :, :, 8:]
        y = nn.functional.pad(circ_ch, pad=self.pad, mode='circular')
        y = torch.cat((y, other_ch), dim=-1)
        # print(f'PRE-CONV: -> {y.shape} \n {y[0][0]}')
        y = self.conv_no_pad(y)
        return y


# %% Convolution to reduce parameter number
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, depth, kernel_size, bias=False, device=None):
        super(SeparableConv2d, self).__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels * depth, kernel_size=kernel_size, groups=in_channels,
                                   padding=1, bias=bias, device=device)  # padding_mode='circular'
        # self.pad_H_zero = nn.ZeroPad2d((0, 0, 1, 1))
        self.pointwise = nn.Conv2d(in_channels * depth, out_channels, kernel_size=1, bias=bias, device=device)

    def forward(self, x):
        y = self.depthwise(x)  # self.depthwise.weight.T.T, self.depthwise.bias)
        # print('Depthwise: ', y.shape)
        # y = self.pad_H_zero(y)
        # print('AFTER ZERO PAD: ', y.shape)
        y = self.pointwise(y)  # self.pointwise.weight.T.T, self.pointwise.bias)
        # print('Pointwise: ', y.shape)
        return y


# %% Standard model for high frequency databases [1000-2000] Hz
class MultiKernelConv2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes=None):
        """
        :param in_channels: 1 for signals
        :param out_channels: will be then multiplied by 2 in the separable conv block -> out total = 2*out_ch
        :param kernel_sizes: need to be an 2D array or tensor with shape (int,2) for 2d convolution, first dimensions
                                sets number of parallel kernel
        """
        super(MultiKernelConv2D, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = np.full((5, 2), [1, 3]).astype(int)
        for i in range(5):
            kernel_sizes[i][0] = 10 * (i + 1)
        if kernel_sizes.shape[1] != 2:
            print("Error, second dimensions must be 2 for convolution")

        self.padd_list = list()
        for j in range(kernel_sizes.shape[0]):
            if kernel_sizes[j][0] % 2 == 0:
                self.padd_list.append((int(kernel_sizes[j][0] / 2), int(kernel_sizes[j][0] / 2) - 1))
            else:
                self.padd_list.append((int((kernel_sizes[j][0] - 1) / 2), int((kernel_sizes[j][0] - 1) / 2)))

        self.kernels = nn.ModuleList(
            nn.Sequential(
                # nn.Conv2d(in_channels, out_channels, tuple(kernel_sizes[k]),
                #           padding=(0, round(kernel_sizes[k][1] / 2)), padding_mode='circular'),  # -> remeber to change
                Conv2d_Circular8(in_channels, out_channels, tuple(kernel_sizes[k]),
                                 pad_LRTB=(round(kernel_sizes[k][1] / 2), round(kernel_sizes[k][1] / 2), 0, 0)),
                nn.BatchNorm2d(out_channels),  # ATTENTION! -> kernel_sizes[:][1] MUST BE ODD
                nn.ZeroPad2d((0, 0, self.padd_list[k][0], self.padd_list[k][1])),  # -> zero padding along timeloa
                nn.ELU(alpha=1.0),
                nn.MaxPool2d((20, 1)),
                # ATTENTION! -> change this MaxPool for different windows dimensions (10,1) for 200 ms
                nn.Dropout2d(0.2),
                SeparableConv2d(out_channels, 2 * out_channels, 1, 3),  # Expanding feature space (out = 2* input)
                nn.BatchNorm2d(2 * out_channels),
                nn.ELU(alpha=1.0),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.2))
            for k in range(np.shape(kernel_sizes)[0]))

    def forward(self, inputs):
        # now you can build a single output from the list of convolutions
        out = [module(inputs) for module in self.kernels]  # -> uncomment to get concatenated
        # for module in self.kernels:
        #     out = [module(inputs)]
        #     outcheck = inputs
        #     print('\nNew Kernel - Module\n')
        #     for seq in module:
        #         print('Input: ', outcheck.shape)
        #         outcheck = seq.forward(outcheck)
        #         print("module: ", seq, "\noutcheck: ", outcheck.shape,'\n\n')
        # you can return a list, or even build a single tensor, like so

        return torch.cat(out, dim=1)


class MKCNN(nn.Module):
    def __init__(self, out_ch: np.array, in_ch=1, multi_kernel_sizes=None, kernel_dim=np.array([1, 3, 3]),
                 number_of_classes=14):
        super(MKCNN, self).__init__()
        """
        :param in_ch:               for signals should be 1
        :param out_ch:              need to be 4 int values array for channels of four conv
        :param multi_kernel_sizes:  need to be an 2D array or tensor with shape (int,2) for all four stages of 2d convolution
        :param kernel_dim:          normal kernel dimension for 3 conv stages
        :param number_of_classes:   Number of classes to classify (output of last FC)
        """
        if multi_kernel_sizes is not None:
            self.multi_kernel_sizes = multi_kernel_sizes
        else:
            multi_kernel_sizes = np.full((5, 2), [1, 3]).astype(int)
            for i in range(5):
                multi_kernel_sizes[i][0] = 10 * (i + 1)
            self.multi_kernel_sizes = multi_kernel_sizes

        self.conv_kernel_dim = kernel_dim
        if out_ch.size != 4:
            raise Exception("There are 4 convolutions, out_ch must have one dim with size 4")
        if multi_kernel_sizes.shape[1] != 2:
            raise Exception("Error, second dimension of kernel_dim needs to be 2 for Conv2D")
        # if kernel_dim.shape[1] != 2:
        #     raise Exception("Error, second dimension of kernel_dim needs to be 2 for Conv2D")
        if kernel_dim.shape[0] != 3:
            raise Exception("Number of conv layer with variable filter size is 3, first dimensions must have length 3")

        self.model = nn.Sequential(
            MultiKernelConv2D(in_ch, out_ch[0], multi_kernel_sizes),
            nn.Conv2d(2 * out_ch[0] * multi_kernel_sizes.shape[0], out_ch[1], kernel_dim[0]),  # Compacting features
            # nn.Conv2d(2* out* MK-sizes bc of MultiKernel is build to double the number of channel/feature
            nn.ELU(alpha=1.0),
            # There is no batch, pooling or dropout here between this stage and the one before, weird!
            SeparableConv2d(out_ch[1], out_ch[2], 1, kernel_dim[1]),
            nn.BatchNorm2d(out_ch[2]),
            nn.ELU(alpha=1.0),
            nn.AdaptiveMaxPool2d((5, 3)),
            # nn.MaxPool2d(2),  # -> modified bc of dimensions reduction with 100Hz
            nn.Dropout2d(0.2),
            SeparableConv2d(out_ch[2], out_ch[3], 1, kernel_dim[2]),
            nn.BatchNorm2d(out_ch[3]),
            nn.ELU(alpha=1.0),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Flatten(),
            # nn.Linear(256, 512),  # out_ch[3] * [windows restricted]
            # nn.ELU(alpha=1.0),
            # nn.BatchNorm1d(512),
            nn.Linear(256, 128),
            nn.ELU(alpha=1.0),
            # nn.BatchNorm1d(128),
            nn.Linear(128, number_of_classes)
        )

    def forward(self, inputs):
        # now you can build a single output from the list of convolutions
        linear1_output = None
        for module in self.model:
            # print("Inputs: ", inputs.shape, "\nmodule: ", module, )
            inputs = module.forward(inputs)
            # print('Output: ', inputs.shape, '\n\n')
            if isinstance(module, nn.Linear) and linear1_output is None:
                linear1_output = inputs
        # you can return a list, or even build a single tensor, like so
        return inputs, linear1_output


# %% Standard model for DB1 or low frequency based databases
class MultiKernelConv2D_20x10(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes=None):
        """
        :param in_channels: 1 for signals
        :param out_channels: will be then multiplied by 2 in the separable conv block -> out total = 2*out_ch
        :param kernel_sizes: need to be an 2D array or tensor with shape (int,2) for 2d convolution, first dimensions
                                sets number of parallel kernel
        """
        super(MultiKernelConv2D_20x10, self).__init__()

        if kernel_sizes is None:
            kernel_sizes = np.full((5, 2), [1, 3]).astype(int)
            for i in range(5):
                kernel_sizes[i][0] = (i + 1)
        if kernel_sizes.shape[1] != 2:
            print("Error, second dimensions must be 2 for convolution2D")

        self.padd_list = list()
        for j in range(kernel_sizes.shape[0]):
            if kernel_sizes[j][0] % 2 == 0:
                self.padd_list.append((int(kernel_sizes[j][0] / 2), int(kernel_sizes[j][0] / 2) - 1))
            else:
                self.padd_list.append((int((kernel_sizes[j][0] - 1) / 2), int((kernel_sizes[j][0] - 1) / 2)))

        self.kernels = nn.ModuleList(
            nn.Sequential(
                # nn.Conv2d(in_channels, out_channels, tuple(kernel_sizes[k]),
                #                 #           padding=(0, int(kernel_sizes[k][1] / 2)), padding_mode='circular'),  # -> remember to change
                Conv2d_Circular8(in_channels, out_channels, tuple(kernel_sizes[k]),
                                 pad_LRTB=(round(kernel_sizes[k][1] / 2), round(kernel_sizes[k][1] / 2), 0, 0)),
                nn.BatchNorm2d(out_channels),  # ATTENTION! -> kernel_sizes[:][1] MUST BE ODD
                nn.ZeroPad2d((0, 0, self.padd_list[k][0], self.padd_list[k][1])),  # -> zero padding along time
                nn.ELU(alpha=1.0),
                # nn.MaxPool2d((2, 1)),  # ATTENTION! -> change this MaxPool for different windows dimensions
                nn.Dropout2d(0.2),
                SeparableConv2d(out_channels, 2 * out_channels, 1, 3),  # Expanding feature space (out = 2* input)
                nn.BatchNorm2d(2 * out_channels),
                nn.ELU(alpha=1.0),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.2))
            for k in range(np.shape(kernel_sizes)[0]))

    def forward(self, inputs):
        # now you can build a single output from the list of convolutions
        out = [module(inputs) for module in self.kernels]  # -> uncomment to get concatenated
        # for module in self.kernels:
        #     out = [module(inputs)]
        #     outcheck = inputs
        #     print('\nNew Kernel - Module\n')
        #    for seq in module:
        #         # print('Input: ', outcheck.shape)
        #         outcheck = seq.forward(outcheck)
        # print("module: ", seq, "\noutcheck: ", outcheck.shape,'\n\n')

        return torch.cat(out, dim=1)

class MKCNN_20x10(nn.Module):
    def __init__(self, out_ch: np.array, in_ch=1, multi_kernel_sizes=None, kernel_dim=np.array([1, 3, 3]),
                 number_of_classes=14):
        super(MKCNN_20x10, self).__init__()
        """
        :param in_ch:               for signals should be 1
        :param out_ch:              need to be 4 int values array for channels of four conv
        :param multi_kernel_sizes:  need to be an 2D array or tensor with shape (int,2) for all four stages of 2d convolution
        :param kernel_dim:          normal kernel dimension for 3 conv stages
        :param number_of_classes:   Number of classes to classify (output of last FC)
        """
        if multi_kernel_sizes is not None:
            self.multi_kernel_sizes = multi_kernel_sizes
        else:
            multi_kernel_sizes = np.full((5, 2), [1, 3]).astype(int)
            for i in range(5):
                multi_kernel_sizes[i][0] = (i + 1)
            self.multi_kernel_sizes = multi_kernel_sizes

        self.conv_kernel_dim = kernel_dim
        if out_ch.size != 4:
            raise Exception("There are 4 convolutions, out_ch must have one dim with size 4")
        if multi_kernel_sizes.shape[1] != 2:
            raise Exception("Error, second dimension of kernel_dim needs to be 2 for Conv2D")
        if kernel_dim.shape[0] != 3:
            raise Exception("Number of conv layer with variable filter size is 3, first dimensions must have length 3")

        self.model = nn.Sequential(
            MultiKernelConv2D_20x10(in_ch, out_ch[0], multi_kernel_sizes),
            nn.Conv2d(2 * out_ch[0] * multi_kernel_sizes.shape[0], out_ch[1], kernel_dim[0]),  # Compatting features
            # nn.Conv2d(2* out* MK-sizes bc of MultiKernel is build to double the number of channel/feature
            nn.ELU(alpha=1.0),
            # There is no batch, pooling or dropout here between this stage and the one before weird!
            SeparableConv2d(out_ch[1], out_ch[2], 1, kernel_dim[1]),
            nn.BatchNorm2d(out_ch[2]),
            nn.ELU(alpha=1.0),
            nn.AdaptiveMaxPool2d((5, 3)),  # nn.MaxPool2d(2), -> modified bc of dimensions reduction with 100Hz
            nn.Dropout2d(0.2),
            SeparableConv2d(out_ch[2], out_ch[3], 1, kernel_dim[2]),
            nn.BatchNorm2d(out_ch[3]),
            nn.ELU(alpha=1.0),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Flatten(),
            nn.Linear(256, 128),  # out_ch[3] * [windows restricted]
            # nn.ELU(alpha=1.0),
            # # nn.BatchNorm1d(512),
            # nn.Linear(512, 128),
            # nn.ELU(alpha=1.0),
            # nn.BatchNorm1d(128),
            nn.Linear(128, number_of_classes),  # nn.Linear(128, number_of_classes),
        )

    def forward(self, inputs):
        linear1_output = None
        for module in self.model:
            # print("Inputs: ", inputs.shape, "\nmodule: ", module, )
            inputs = module.forward(inputs)
            if isinstance(module, nn.Linear) and linear1_output is None:
                linear1_output = inputs
            # print('Output: ', inputs.shape, '\n\n')

        return inputs, linear1_output


# %% parametric models from parameter tuning
class MultiKernelConv2D_grid(nn.Module):
    # def __init__(self, in_channels: int, out_channels: int, kernel_sizes=None):
    def __init__(self, dict: dict):

        super(MultiKernelConv2D_grid, self).__init__()
        self.activation_fun = dict['act_func']
        self.ch = dict['N_multik']
        self.kernels = dict['Kernel_multi_dim']
        self.padd_list = list()

        if 'Pool_Type' in dict.keys():
            self.pool_type = dict['Pool_Type']
        else:
            self.pool_type = nn.MaxPool2d

        if 'wnd_len' in dict.keys():
            # Based on the selected wnd_lenght, take the maximum 1/20 part of the temporal dimensions
            self.pool_layer = self.pool_type((int(dict['wnd_len'] / 20), 1))
        else:
            self.pool_layer = self.pool_type(1)

        for j in range(self.kernels.shape[0]):
            if self.kernels[j][0] % 2 == 0:
                self.padd_list.append((int(self.kernels[j][0] / 2), int(self.kernels[j][0] / 2) - 1))
            else:
                self.padd_list.append((int((self.kernels[j][0] - 1) / 2), int((self.kernels[j][0] - 1) / 2)))

        self.towers = nn.ModuleList(
            nn.Sequential(
                # nn.Conv2d(1, self.ch, tuple(self.kernels[k]),
                #           padding=(0, int(self.kernels[k][1] / 2)), padding_mode='circular'),
                # Padding is 'circular' to keep consistency in space (ch) dim (circular shape)
                Conv2d_Circular8(1, self.ch, tuple(self.kernels[k]),
                                 pad_LRTB=(round(self.kernels[k][1] / 2), round(self.kernels[k][1] / 2), 0, 0)),
                nn.BatchNorm2d(self.ch),  # ATTENTION! -> kernel_sizes[:][1] MUST BE ODD FOR EACH TOWER OR RAISES ERROR
                nn.ZeroPad2d((0, 0, self.padd_list[k][0], self.padd_list[k][1])),  # -> zero padding along time
                self.activation_fun,
                self.pool_layer,  
                nn.Dropout2d(0.2),
                SeparableConv2d(self.ch, 2 * self.ch, 1, kernel_size=3),  # Expanding feature space (out = 2* input)
                nn.BatchNorm2d(2 * self.ch),
                self.activation_fun,
                self.pool_type(2))  # nn.Dropout2d(0.5))

            for k in range(np.shape(self.kernels)[0]))

    def forward(self, inputs):
        out = [module(inputs) for module in self.towers]  # -> uncomment to get concatenated

        # To check flow and print outputs
        # out = []
        # for module in self.towers:
        #     outcheck = inputs
        #     print('#' * 100)
        #     print('\nNew Kernel - Module\n')
        #     print('#' * 100)
        #     for seq in module:
        #         # print('Input: ', outcheck.shape)
        #         outcheck = seq.forward(outcheck)
        #         print("module: ", seq, "\noutcheck: ", outcheck.shape, '\n\n')
        #     out.append(outcheck)

        return torch.cat(out, dim=1)


class MKCNN_grid(nn.Module):
    # def __init__(self, out_ch: np.array, in_ch=1, multi_kernel_sizes=None, kernel_dim=np.array([1, 3, 3]),
    #              number_of_classes=11):
    #    super(MKCNN, self).__init__()

    def __init__(self, dict, num_classes=14):
        super(MKCNN_grid, self).__init__()

        self.activation_fun = dict['act_func']
        self.in_ch = dict['N_multik'] * 2 * len(dict['Kernel_multi_dim'])  # Based on concatenation of MultiKernelConv2D

        if 'Pool_Type' in dict.keys():
            self.pool_type = dict['Pool_Type']
        else:
            self.pool_type = nn.MaxPool2d

        if 'wnd_len' in dict.keys():
            self.pool_layer = self.pool_type(2)
        else:
            self.pool_layer = nn.AdaptiveMaxPool2d((5, 3))  # -> because with DB1 we have only 10 channel I adjust it

        self.model = nn.Sequential(
            MultiKernelConv2D_grid(dict),
            nn.Conv2d(self.in_ch, dict['N_Conv_conc'], dict['Kernel_Conv_conc'],
                      padding=int((dict['Kernel_Conv_conc'] - 1) / 2)),
            self.activation_fun,
            # There is no batch, pooling or dropout here between this stage and the one before: weird!
            SeparableConv2d(dict['N_Conv_conc'], dict['N_SepConv'], 1, kernel_size=3),
            nn.BatchNorm2d(dict['N_SepConv']),
            self.activation_fun,
            self.pool_layer,  # nn.MaxPool2d(2), -> modified bc of dimensions reduction wnd_len -> 20x10
            nn.Dropout2d(0.2),
            SeparableConv2d(dict['N_SepConv'], dict['N_SepConv'], 1, kernel_size=3),
            nn.BatchNorm2d(dict['N_SepConv']),
            self.activation_fun,
            self.pool_type(2),
            nn.Dropout2d(0.2),
            nn.Flatten(),
            # nn.Linear(2 * dict['N_SepConv'], 512),  # out_ch[3] * [windows restricted]
            # nn.BatchNorm1d(512),
            # self.activation_fun,
            nn.Linear(2 * dict['N_SepConv'], 128),
            nn.BatchNorm1d(128),
            self.activation_fun,
            nn.Linear(128, num_classes),
            # nn.Softmax(num_classes) # -> not inserted because I calculate the loss during the training & BEFORE the softmax
        )

    def forward(self, inputs):
        # now you can build a single output from the list of convolutions
        linear1_out = None
        for module in self.model:
            # print("Inputs: ", inputs.shape, "\nmodule: ", module )
            inputs = module.forward(inputs)

            # Pick the output of the first linear
            if isinstance(module, nn.Linear) and linear1_out is None:   # Take embeddings of first linear layer
                linear1_out = inputs
            # print('Output: ', inputs.shape, '\n\n')

        return inputs, linear1_out

    def freeze_multik(self):
        for layer in self.model[0].towers:     # Taking all layers from the different towers (e.g. parallel convolution)
            for param in layer.parameters():
                param.requires_grad = False

    def reset(self):
        '''
        -> This function resets the parameters of each layer
           of self (by checking beforehand it is possible)
        '''
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# REVERSAL GRADIENT PARAMETRIC MODEL AND TRAINING
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class MKCNN_grid_AN(nn.Module):

    def __init__(self, dict, num_domains, num_classes=14):
        super(MKCNN_grid_AN, self).__init__()

        self.activation_fun = dict['act_func']
        self.in_ch = dict['N_multik'] * 2 * len(dict['Kernel_multi_dim'])  # Based on concatenation of MultiKernelConv2D
        self.num_domains = num_domains

        if 'Pool_Type' in dict.keys():
            self.pool_type = dict['Pool_Type']
        else:
            self.pool_type = nn.MaxPool2d

        if 'wnd_len' in dict.keys():
            self.pool_layer = self.pool_type(2)
        else:
            self.pool_layer = nn.AdaptiveMaxPool2d((5, 3))  # -> because with DB1 we have only 10 channel I adjust it

        self.model = nn.Sequential(
            MultiKernelConv2D_grid(dict),
            nn.Conv2d(self.in_ch, dict['N_Conv_conc'], dict['Kernel_Conv_conc'],
                      padding=int((dict['Kernel_Conv_conc'] - 1) / 2)),
            self.activation_fun,
            # There is no batch, pooling or dropout here between this stage and the one before: weird!
            SeparableConv2d(dict['N_Conv_conc'], dict['N_SepConv'], 1, kernel_size=3),
            nn.BatchNorm2d(dict['N_SepConv']),
            self.activation_fun,
            self.pool_layer,  # nn.MaxPool2d(2), -> modified bc of dimensions reduction wnd_len -> 20x10
            nn.Dropout2d(0.2),
            SeparableConv2d(dict['N_SepConv'], dict['N_SepConv'], 1, kernel_size=3),
            nn.BatchNorm2d(dict['N_SepConv']),
            self.activation_fun,
            self.pool_type(2),
            nn.Dropout2d(0.2),
            nn.Flatten(),
            # nn.Linear(2 * dict['N_SepConv'], 512),  # out_ch[3] * [windows restricted]
            # nn.BatchNorm1d(512),
            # self.activation_fun,
            nn.Linear(2 * dict['N_SepConv'], 128),
            nn.BatchNorm1d(128),
            self.activation_fun,
            nn.Linear(128, num_classes),
            # nn.Softmax(num_classes) # -> not inserted because I calculate the loss during the training & BEFORE the softmax
        )

        if 'Output2' in dict.keys():
            self.output2 = dict['Output2']

            if self.output2 == nn.Flatten:
                self.domain_classifier = nn.Sequential(
                    nn.Linear(2 * dict['N_SepConv'], 128),
                    self.activation_fun,
                    nn.Linear(128, self.num_domains))

            elif self.output2 == MultiKernelConv2D_grid:
                self.domain_classifier = nn.Sequential(
                    nn.Conv2d(self.in_ch, dict['N_Conv_conc'], dict['Kernel_Conv_conc'],
                              padding=int((dict['Kernel_Conv_conc'] - 1) / 2)),
                    self.activation_fun,
                    # There is no batch, pooling or dropout here between this stage and the one before: weird!
                    SeparableConv2d(dict['N_Conv_conc'], dict['N_SepConv'], 1, kernel_size=3),
                    nn.BatchNorm2d(dict['N_SepConv']),
                    self.activation_fun,
                    self.pool_layer,  # nn.MaxPool2d(2), -> modified bc of dimensions reduction wnd_len -> 20x10
                    nn.Dropout2d(0.2),
                    SeparableConv2d(dict['N_SepConv'], dict['N_SepConv'], 1, kernel_size=3),
                    nn.BatchNorm2d(dict['N_SepConv']),
                    self.activation_fun,
                    self.pool_type(2),
                    nn.Dropout2d(0.2),
                    nn.Flatten(),
                    nn.Linear(2* dict['N_SepConv'], 128),
                    nn.BatchNorm1d(128),
                    self.activation_fun,
                    nn.Linear(128, self.num_domains))

                # self.domain_classifier = nn.Sequential( 
                #     nn.Conv2d(self.in_ch, dict['N_Conv_conc'], dict['Kernel_Conv_conc'],
                #               padding=int((dict['Kernel_Conv_conc'] - 1) / 2)),                                               
                #     nn.Flatten(),
                #     nn.Linear(dict['N_Conv_conc'] * 70, 512),
                #     nn.Linear(512, 128),
                #     nn.Linear(128, self.num_domains))
        else:
            raise ValueError('Module not found for second output!')

    def forward(self, inputs, alpha=None):
        # now you can build a single output from the list of convolutions
        output2 = None
        for module in self.model:
            # print("Inputs: ", inputs.shape, "\nmodule: ", module, )
            # print(f'INPUTS: {inputs.shape}        MODULE: -> {module}')
            inputs = module.forward(inputs)
            if alpha is not None:
                if isinstance(module, self.output2) and output2 is None:
                    # dims = inputs.size()
                    # print(f'SIZE: {dims}')
                    rev_feature = GradientReversalLayer.apply(inputs, alpha)
                    # print(f'REVERSED OUT: {rev_feature}')

                    output2 = self.domain_classifier(rev_feature)
                # print('Output: ', inputs.shape, '\n\n')

        return inputs, output2

    def freeze_multik(self):
        for layer in self.model[0].towers:
            for param in layer.parameters():
                param.requires_grad = False

    def initialize_domain_class_weights(self):
        for layer in self.domain_classifier:
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, SeparableConv2d):
                nn.init.xavier_normal_(layer.depthwise.weight)
                nn.init.xavier_normal_(layer.pointwise.weight)
                if layer.depthwise.bias is not None:
                    nn.init.constant_(layer.depthwise.bias, 0)
                if layer.pointwise.bias is not None:
                    nn.init.constant_(layer.pointwise.bias, 0)
                    
    def reinitialize_last_linear_layers(self):
        # Identify and re-initialize the last two linear layers
        linear_layers = [layer for layer in self.model if isinstance(layer, nn.Linear)]
        for layer in linear_layers[-2:]:
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)



# %% Training types
def train_model_standard(model, loss_fun, optimizer, dataloaders, scheduler, num_epochs=100, precision=1e-8,
                         patience=10, patience_increase=10, device=None, l2=None):
    if not device:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device("cpu")

    since = time.time()
    tr_epoch_loss_vec = []
    val_epoch_loss_vec = []
    # best_loss = float('inf')
    best_loss = 10000000.0

    softmax_block = nn.Softmax(dim=1)
    best_state = {'epoch': 0, 'state_dict': copy.deepcopy(model.state_dict()),
                  'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(70 * '#', 5 * '\n', 'Epoch {}/{}'.format(epoch, num_epochs - 1), 70 * '-', 5 * '\n')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            tot_batch = len(dataloaders[phase])
            total = 0
            cc = 0
            loss_accumulator = 0.0

            y_true = []
            y_pred = []

            for inputs, labels in dataloaders[phase]:
                cc = cc + 1

                inputs = inputs[:, None, :, :]  # Adding on
                inputs = inputs.to(device)

                labels = labels.type(torch.LongTensor).to(device)  # Loss_Fun want LongTensor
                labels_np = labels.cpu().data.numpy()

                if phase == 'train':
                    model.train()

                    outputs, out2 = model.forward(inputs)
                    outputs_softmax = softmax_block(outputs)

                    outputs_np = outputs_softmax.cpu().data.numpy()
                    outputs_np = np.argmax(outputs_np, axis=1)

                    y_true = np.append(y_true, outputs_np)
                    y_pred = np.append(y_pred, labels_np)

                    if l2:
                        loss = loss_fun(outputs, labels) + l2
                    else:
                        loss = loss_fun(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    model.zero_grad()

                else:
                    model.eval()
                    with torch.no_grad():
                        # forward
                        outputs, _ = model(inputs)

                        outputs_softmax = softmax_block(outputs)
                        outputs_np = outputs_softmax.cpu().data.numpy()
                        outputs_np = np.argmax(outputs_np, axis=1)

                        y_true = np.append(y_true, outputs_np)
                        y_pred = np.append(y_pred, labels_np)

                        loss = loss_fun(outputs, labels)

                loss_accumulator = loss_accumulator + ((1 / (total + 1)) * (loss.item() - loss_accumulator))
                total = total + 1

                acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
                kappa = metrics.cohen_kappa_score(y1=y_true, y2=y_pred, weights='quadratic')
                print(phase + " -> loss: {:0.5f}".format(loss_accumulator) + ", accuracy: {:0.5f}".format(acc) +
                      ", kappa score: {:0.4f}".format(
                          kappa) + f', Epoch: -> {epoch} / {num_epochs}  Batch Number: -> {cc} / {tot_batch}')

            if phase == 'train':
                tr_epoch_loss_vec.append(loss_accumulator)
            elif phase == 'val':
                val_epoch_loss_vec.append(loss_accumulator)
            else:
                raise ValueError('Dictionary with dataloaders has a phase different from "train" or "val"')

            # deep copy the model
            if phase == 'val':
                # scheduler.step(loss_accumulator)
                if loss_accumulator + precision < best_loss:
                    print(70 * '#', 5 * '\n', "New best validation loss:", loss_accumulator)
                    best_loss = loss_accumulator
                    best_state = {'epoch': epoch + 1, 'state_dict': copy.deepcopy(model.state_dict()),
                                  'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
                    patience = patience_increase + epoch
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - epoch_start), 70 * '#',
              5 * '\n')
        if epoch > patience:
            break

    tr_epoch_loss_vec, val_epoch_loss_vec = np.array(tr_epoch_loss_vec), np.array(val_epoch_loss_vec)
    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    return best_state, tr_epoch_loss_vec, val_epoch_loss_vec


def pre_train_model_triplet(model, loss_fun, optimizer, dataloaders, scheduler, num_epochs=100, precision=1e-8,
                            patience=10, patience_increase=10, device=None):
    if not device:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device("cpu")

    since = time.time()
    tr_epoch_loss_vec = []
    val_epoch_loss_vec = []
    best_loss = float('inf')

    softmax_block = nn.Softmax(dim=1)
    best_state = {'epoch': 0, 'state_dict': copy.deepcopy(model.state_dict()),
                  'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(70 * '#', 5 * '\n', 'Epoch {}/{}'.format(epoch, num_epochs - 1), 70 * '-', 5 * '\n')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            tot_batch = len(dataloaders[phase])
            total = 0
            cc = 0
            loss_accumulator = 0.0

            y_true = []
            y_pred = []

            for inputs, labels, pos, neg in dataloaders[phase]:
                cc = cc + 1
                # inputs = torch.swapaxes(inputs, 2, 1)  # -> convert from [10,20] to [20,10]
                inputs, pos, neg = inputs[:, None, :, :], pos[:, None, :, :], neg[:, None, :, :]
                inputs, pos, neg = inputs.to(device), pos.to(device), neg.to(device)
                labels_np = labels.cpu().data.numpy()

                if phase == 'train':

                    model.train()
                    # forward
                    outputs, anchor = model.forward(inputs)
                    _, out_pos = model.forward(pos)
                    _, out_neg = model.forward(neg)

                    outputs_softmax = softmax_block(outputs)
                    outputs_np = outputs_softmax.cpu().data.numpy()
                    outputs_np = np.argmax(outputs_np, axis=1)

                    y_true = np.append(y_true, outputs_np)
                    y_pred = np.append(y_pred, labels_np)

                    loss = loss_fun(anchor, out_pos, out_neg)
                    loss.backward()
                    optimizer.step()

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    model.zero_grad()

                else:
                    model.eval()
                    with torch.no_grad():

                        # forward
                        outputs, anchor = model(inputs)
                        _, outs_pos = model(pos)
                        _, outs_neg = model(neg)

                        outputs_softmax = softmax_block(outputs)
                        outputs_np = outputs_softmax.cpu().data.numpy()
                        outputs_np = np.argmax(outputs_np, axis=1)

                        y_true = np.append(y_true, outputs_np)
                        y_pred = np.append(y_pred, labels_np)

                        loss = loss_fun(anchor, outs_pos, outs_neg)

                loss_accumulator = loss_accumulator + ((1 / (total + 1)) * (loss.item() - loss_accumulator))
                total = total + 1

                acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
                kappa = metrics.cohen_kappa_score(y1=y_true, y2=y_pred, weights='quadratic')
                print(phase + " -> Loss: {:0.6f}".format(loss_accumulator) + "  Accuracy: {:0.6f}".format(acc) +
                      "  Kappa score: {:0.4f}".format(
                          kappa) + f'  Epoch: -> {epoch} / {num_epochs}  Batch Number: -> {cc} / {tot_batch}')

            if phase == 'train':
                tr_epoch_loss_vec.append(loss_accumulator)
            elif phase == 'val':
                val_epoch_loss_vec.append(loss_accumulator)
            else:
                raise ValueError('Something unexpected occurred')

            # deep copy the model
            if phase == 'val':
                # scheduler.step(loss_accumulator)
                if loss_accumulator + precision < best_loss:
                    print(70 * '#', 5 * '\n', "New best validation loss:", loss_accumulator)
                    best_loss = loss_accumulator
                    best_state = {'epoch': epoch + 1, 'state_dict': copy.deepcopy(model.state_dict()),
                                  'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
                    patience = patience_increase + epoch
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - epoch_start), 70 * '#',
              5 * '\n')
        if epoch > patience:
            break

    tr_epoch_loss_vec, val_epoch_loss_vec = np.array(tr_epoch_loss_vec), np.array(val_epoch_loss_vec)
    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    return best_state, tr_epoch_loss_vec, val_epoch_loss_vec


def train_model_triplet(model, loss_fun, optimizer, dataloaders, scheduler, num_epochs=100, precision=1e-8,
                        patience=10, patience_increase=10, device=None, margin=1, p_dist=2, beta=1):
    if not device:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device("cpu")

    since = time.time()
    tr_epoch_loss_vec = []
    val_epoch_loss_vec = []
    best_loss = float('inf')

    softmax_block = nn.Softmax(dim=1)
    triplet_loss = nn.TripletMarginLoss(margin=margin, p=p_dist, reduction='mean')

    best_state = {'epoch': 0, 'state_dict': copy.deepcopy(model.state_dict()),
                  'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(70 * '#', 5 * '\n', 'Epoch {}/{}'.format(epoch, num_epochs - 1), 70 * '-', 5 * '\n')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            tot_batch = len(dataloaders[phase])
            total = 0
            cc = 0
            loss_accumulator = 0.0

            y_true = []
            y_pred = []

            for inputs, labels, pos, neg in dataloaders[phase]:
                cc = cc + 1
                # imposing one channel to the window [batch, ch, width, height]
                inputs, pos, neg = inputs[:, None, :, :], pos[:, None, :, :], neg[:, None, :, :]
                inputs, pos, neg = inputs.to(device), pos.to(device), neg.to(device)

                labels = labels.type(torch.LongTensor).to(device)  # Loss_Fun want LongTensor
                labels_np = labels.cpu().data.numpy()

                if phase == 'train':
                    model.train()
                    # forward
                    # print('Inputs: ', inputs)
                    outputs, anchor = model(inputs)
                    _, outs_pos = model.forward(pos)
                    _, outs_neg = model.forward(neg)

                    outputs_softmax = softmax_block(outputs)
                    outputs_np = outputs_softmax.cpu().data.numpy()
                    outputs_np = np.argmax(outputs_np, axis=1)

                    y_true = np.append(y_true, outputs_np)
                    y_pred = np.append(y_pred, labels_np)

                    loss_cl = loss_fun(outputs, labels)
                    loss_triplet = triplet_loss(anchor, outs_pos, outs_neg)
                    loss = loss_cl + beta * loss_triplet
                    loss.backward()
                    optimizer.step()
                    # loss = loss.item()
                    optimizer.zero_grad()
                    model.zero_grad()

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    model.zero_grad()

                else:
                    model.eval()
                    with torch.no_grad():

                        # forward
                        outputs, _ = model(inputs)

                        outputs_softmax = softmax_block(outputs)
                        outputs_np = outputs_softmax.cpu().data.numpy()
                        outputs_np = np.argmax(outputs_np, axis=1)

                        y_true = np.append(y_true, outputs_np)
                        y_pred = np.append(y_pred, labels_np)

                        loss = loss_fun(outputs, labels)

                loss_accumulator = loss_accumulator + ((1 / (total + 1)) * (loss.item() - loss_accumulator))
                total = total + 1

                acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
                kappa = metrics.cohen_kappa_score(y1=y_true, y2=y_pred, weights='quadratic')
                print(phase + " -> loss: {:0.6f}".format(loss_accumulator) + ", accuracy: {:0.6f}".format(acc) +
                      ", kappa score: {:0.4f}".format(
                          kappa) + f', Epoch: -> {epoch} / {num_epochs}  Batch Number: -> {cc} / {tot_batch}')

            if phase == 'train':
                tr_epoch_loss_vec.append(loss_accumulator)
            elif phase == 'val':
                val_epoch_loss_vec.append(loss_accumulator)
            else:
                raise ValueError('Something unexpected occurred')

            # deep copy the model
            if phase == 'val':
                # scheduler.step(loss_accumulator)
                if loss_accumulator + precision < best_loss:
                    print(70 * '#', 5 * '\n', "New best validation loss:", loss_accumulator)
                    best_loss = loss_accumulator
                    best_state = {'epoch': epoch + 1, 'state_dict': copy.deepcopy(model.state_dict()),
                                  'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
                    patience = patience_increase + epoch
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - epoch_start), 70 * '#',
              5 * '\n')
        if epoch > patience:
            break

    tr_epoch_loss_vec, val_epoch_loss_vec = np.array(tr_epoch_loss_vec), np.array(val_epoch_loss_vec)
    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    return best_state, tr_epoch_loss_vec, val_epoch_loss_vec

# Function to train the model
def train_model_reversal_gradient(model, loss_fun, optimizer, dataloaders, scheduler, lamba=0.5, alpha_start=0,
                                  num_epochs=100, precision=1e-8, loss_fun_domain=nn.CrossEntropyLoss(),
                                  patience=10, patience_increase=10, device=None):
    if not device:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device("cpu")

    since = time.time()
    tr_epoch_loss = []
    val_epoch_loss = []
    tr_epoch_loss_domain = []
    val_epoch_loss_domain = []
    tr_epoch_loss_task = []
    val_epoch_loss_task = []

    # best_loss = float('inf')
    best_loss = 10000000.0

    softmax_block = nn.Softmax(dim=1)
    best_state = {'epoch': 0, 'state_dict': copy.deepcopy(model.state_dict()),
                  'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}

    # Inserted to check value alpha
    alpha_vec = []

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(70 * '#', 5 * '\n', 'Epoch {}/{}'.format(epoch, num_epochs - 1), 70 * '-', 5 * '\n')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print(70 * '#', 5 * '\n', f'PHASE: -> {phase}', 5 * '\n', 70 * '#')
            tot_batch = len(dataloaders[phase])
            total = 0
            cc = 0
            loss_accumulator = 0.0
            domain_loss_acc = 0.0
            task_loss_acc = 0.0
            y_true = []
            y_pred = []

            p = float(epoch * tot_batch) / num_epochs / tot_batch

            alpha = alpha_start + 2. / (1. + np.exp(-10 * p)) - 1
            # Inserted to check values
            alpha_vec.append(alpha)
            print(70 * '#', 5 * '\n', f'ALPHA_VEC: -> {alpha_vec}', 5 * '\n', 70 * '#')

            for inputs, labels, sub in dataloaders[phase]:
                cc = cc + 1

                inputs = inputs[:, None, :, :]  # Adding on
                inputs = inputs.to(device)

                labels = labels.type(torch.LongTensor).to(device)  # Loss_Fun want LongTensor
                labels_np = labels.cpu().data.numpy()

                sub = sub.type(torch.LongTensor).to(device)
                # print(f'LABEL: {labels[0]}      SUB:{sub[0]}')
                if phase == 'train':
                    model.train()

                    outputs, out_domain = model.forward(inputs, alpha)
                    # print(f'O1: {outputs.shape}     O2: {out_domain.shape}')
                    outputs_softmax = softmax_block(outputs)

                    outputs_np = outputs_softmax.cpu().data.numpy()
                    outputs_np = np.argmax(outputs_np, axis=1)

                    y_true = np.append(y_true, outputs_np)
                    y_pred = np.append(y_pred, labels_np)

                    loss_task = loss_fun(outputs, labels)
                    loss_domain = loss_fun_domain(out_domain, sub)
                    loss = loss_task + (lamba * loss_domain)
                    loss.backward()
                    optimizer.step()

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    model.zero_grad()


                else:
                    model.eval()
                    with torch.no_grad():
                        # forward
                        outputs, out_domain = model(inputs, alpha)

                        outputs_softmax = softmax_block(outputs)
                        outputs_np = outputs_softmax.cpu().data.numpy()
                        outputs_np = np.argmax(outputs_np, axis=1)

                        y_true = np.append(y_true, outputs_np)
                        y_pred = np.append(y_pred, labels_np)

                        loss_task = loss_fun(outputs, labels)
                        loss_domain = loss_fun(out_domain, sub)
                        loss = loss_task + (lamba * loss_domain)

                loss_accumulator = loss_accumulator + ((1 / (total + 1)) * (loss.item() - loss_accumulator))
                domain_loss_acc = domain_loss_acc + ((1 / (total + 1)) * (loss_domain.item() - domain_loss_acc))
                task_loss_acc = task_loss_acc + ((1 / (total + 1)) * (loss_task.item() - task_loss_acc))
                total = total + 1

                acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
                kappa = metrics.cohen_kappa_score(y1=y_true, y2=y_pred, weights='quadratic')
                print(phase + " -> loss: {:0.5f}".format(loss_accumulator) + ", accuracy: {:0.5f}".format(acc) +
                      ", kappa score: {:0.4f}".format(
                          kappa) + f', Epoch: -> {epoch} / {num_epochs}  Batch Number: -> {cc} / {tot_batch}')

            if phase == 'train':
                tr_epoch_loss.append(loss_accumulator)
                tr_epoch_loss_domain.append(domain_loss_acc)
                tr_epoch_loss_task.append(task_loss_acc)
            elif phase == 'val':
                val_epoch_loss.append(loss_accumulator)
                val_epoch_loss_domain.append(domain_loss_acc)
                val_epoch_loss_task.append(task_loss_acc)
            else:
                raise ValueError('Dictionary with dataloaders has a phase different from "train" or "val"')

            # deep copy the model
            if phase == 'val':
                # scheduler.step(loss_accumulator)
                if loss_accumulator + precision < best_loss:
                    print(70 * '#', 5 * '\n', "New best validation loss:", loss_accumulator)
                    best_loss = loss_accumulator
                    best_state = {'epoch': epoch + 1, 'state_dict': copy.deepcopy(model.state_dict()),
                                  'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
                    patience = patience_increase + epoch
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - epoch_start), 70 * '#',
              5 * '\n')
        if epoch > patience:
            break

    tr_epoch_loss, val_epoch_loss = np.array(tr_epoch_loss), np.array(val_epoch_loss)
    tr_epoch_loss_domain, val_epoch_loss_domain = np.array(tr_epoch_loss_domain), np.array(val_epoch_loss_domain)
    tr_epoch_loss_task, val_epoch_loss_task = np.array(tr_epoch_loss_task), np.array(val_epoch_loss_task)
    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    return best_state, tr_epoch_loss, val_epoch_loss, tr_epoch_loss_domain, val_epoch_loss_domain, tr_epoch_loss_task, val_epoch_loss_task


#%% L2 NORM
def get_l2_regularization(model, lambda_l2):
    l2_reg = 0.0
    for param in model.parameters():
        l2_reg += torch.norm(param, p=2).pow(2)
    l2_reg *= lambda_l2
    return l2_reg

#%% TCN_CBAM
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0.05, act_fun = None):
        super(TCNBlock, self).__init__()

        # self.conv = Conv2d_Circular8(in_channels, out_channels, kernel_size, stride=stride, pad_LRTB=(1, 1, 0, 0))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        if act_fun:
            self.act_fun = act_fun
        else: 
            self.act_fun = nn.ReLU() 

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.act_fun(x)
        x = self.dropout(x)
        return x


# %% Channel spatial
class FCAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), pool_kernel=(4, 12)):
        super(FCAM, self).__init__()
        self.avg_p = nn.Sequential(
            nn.AvgPool2d(kernel_size=pool_kernel),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=kernel_size))
        self.max_p = nn.Sequential(
            nn.MaxPool2d(kernel_size=(pool_kernel)),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=kernel_size))

    def forward(self, x):
        avg_p = self.avg_p(x)
        max_p = self.max_p(x)
        x = avg_p + max_p
        x = nn.Sigmoid()(x)
        return x


# %% Feature spatial attention
class FSAM(nn.Module):
    def __init__(self, kernel_size=(7, 7)):
        super(FSAM, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=3)

    def forward(self, x):
        avg_p = torch.mean(x, dim=1, keepdim=True)
        max_p, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat(tensors=(max_p, avg_p), dim=1)
        x = self.conv(x)
        x = nn.Sigmoid()(x)
        return x


# %% FINAL MODEL
class TCN_ch_sp_ATT(nn.Module):
    def __init__(self, wnd_len, n_classes, in_ch=1, out_ch=32, tcn_kernels=None, drop=0.5, l2_lambda=None, 
                 dom_out = None, num_domains = None, act_fun = None):
        super(TCN_ch_sp_ATT, self).__init__()

        if tcn_kernels is None:
            tcn_kernels = [3, 7, 7]
            
        if act_fun:
            self.act_fun = act_fun
        else: 
            self.act_fun = nn.ReLU()  

        self.l2_lambda = l2_lambda
        self.avg = round(wnd_len / 5)
        self.TCN = nn.Sequential(
            TCNBlock(in_channels=in_ch, out_channels=out_ch, kernel_size=(tcn_kernels[0], 1), dropout=drop),
            TCNBlock(in_channels=out_ch, out_channels=2 * out_ch, kernel_size=(tcn_kernels[1], 1), dropout=drop),
            TCNBlock(in_channels=2 * out_ch, out_channels=4 * out_ch, kernel_size=(tcn_kernels[2], 1), dropout=drop),
        )
        # self.avg1 = nn.AvgPool2d(kernel_size=(self.avg, 1))
        self.avg1 = nn.AdaptiveAvgPool2d(output_size=((4, 12)))
        self.avg2 = nn.Sequential(
            nn.Conv2d(in_channels=4 * out_ch, out_channels=4 * out_ch, kernel_size=(1, 1), padding=0),
            nn.BatchNorm2d(4 * out_ch), 
            self.act_fun,
            # nn.AvgPool2d(kernel_size=(self.avg, 1)))
            nn.AdaptiveAvgPool2d(output_size=(4, 12)))
        self.avg3 = nn.Sequential(
            nn.Conv2d(in_channels=4 * out_ch, out_channels=4 * out_ch, kernel_size=(3, 3), padding=1),
            # # I use the circular padding over channel axis to keep spatial relation between them
            # Conv2d_Circular8(in_channels=4 * out_ch, out_channels=4 * out_ch, kernel_size=(3, 3),
            #                  pad_LRTB=(1, 1, 0, 0)),
            nn.BatchNorm2d(4 * out_ch),
            self.act_fun,
            # nn.AvgPool2d(kernel_size=(self.avg, 1)))
            nn.AdaptiveAvgPool2d(output_size=(4, 12)))

        self.FCAM = FCAM(in_channels= 384, out_channels=48)
        self.FSAM = FSAM()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(1, 1)),
            nn.BatchNorm2d(384),
            self.act_fun,
            nn.Flatten(),
            nn.Dropout(drop),
            nn.Linear(384 * 4 * 12, 256),
            nn.Dropout(drop/2),
            nn.Linear(256, 128),
            nn.Linear(128, n_classes)
        )

        # Here it possible to decide if forcing the first linear layer to be domain-invariant (output2 == nn.Linear) or no.
        self.output2 = dom_out
        if self.output2 == nn.Flatten:
            self.domain_classifier = nn.Sequential(
                nn.Linear(384 * 4 * 12, 256),
                nn.Linear(256, 128),
                nn.Linear(128, num_domains)
            )
        elif self.output2 == nn.Linear:
            self.domain_classifier = nn.Sequential(
                nn.Linear(256, 128),
                nn.Linear(128, num_domains)
            )

    def forward(self, x, alpha = None):

        for layer in self.TCN:
            # TCN forward
            # print(f'Input: -> {x.shape}')
            x = layer.forward(x)
            # print(f'Module: -> {layer} \n Output: -> {x.shape}')
        # Parallel different average pooling
        x1, x2, x3, = self.avg1(x), self.avg2(x), self.avg3(x)
        x = torch.cat(tensors=(x1, x2, x3), dim=-3)
        # print(f'CONCAT: {x.shape}')
        fcam = self.FCAM(x)
        # print(f'FCAM OUT: {fcam.shape}')
        x = x * fcam
        # print(f'AFTER MULTIPLICATION: {x.shape} -> FSAM INPUT')
        fsam = self.FSAM(x)
        # print(f'FSAM OUTPUT: {fsam.shape} ')
        x = x * fsam
        # print(f'FSAM X FCAM_OUT: {x.shape}')


        out_dom = None
        for layer2 in self.conv_block:
            # print(f'Input: -> {x.shape}')
            x = layer2.forward(x)
            # print(f'Module: -> {layer2} \n Output: -> {x.shape}')
            # l2 = get_l2_regularization(self, lambda_l2=self.l2_lambda)
            if alpha is not None:
                if isinstance(layer2, self.output2):  # the domain classifier can start AFTER the flatten or AFTER the first linear layer
                    # dims = x.size()
                    # print(f'SIZE: {dims}')
                    rev_feature = GradientReversalLayer.apply(x, alpha)
                    # print(f'REVERSED OUT: {rev_feature.shape}')

                    out_dom = self.domain_classifier(rev_feature)
                # print('Output: ', x.shape, '\n\n')

        return x, out_dom

# %% TL results (too much params to pass)

def get_TL_results(model, tested_sub_df, path_to_save, train_rep, valid_rep, test_rep, exe_labels, loss_fn, optimizer,
                    train_type = 'Standard', num_epochs = 10, device = None, scheduler=None, filename=None, **dataloader_param):

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    if device is None:
        device = torch.device('cuda')

    # Check unique sub
    if len(np.unique(tested_sub_df['sub'])) != 1:
        raise ValueError('Zero or More than one subject in Dataframe for TL experiment!')
    sub = tested_sub_df['sub'].values[0]

    if filename is None:
        filename = f'TL_Sub{sub}_on_{len(train_rep)}_Reps'
    else:
        filename = f'{filename}_TL_Sub{sub}_on_{len(train_rep)}_Reps'  # + \
    #    re.split(r'\d+', os.path.basename(os.path.normpath(state_dict_path)))[1][:-4]
    
    test = tested_sub_df[~tested_sub_df['rep'].isin(test_rep)]
    test_set = Pandas_Dataset(test.groupby('sample_index'))
    test_dl = adjust_dataloader(test_set, **dataloader_param)
    
    
    train, valid = None, None
    if train_rep != []:      # If I'm training

        if train_type == 'Standard':
            # Divide dataloader based on number of repetition
            train = tested_sub_df.loc[tested_sub_df['rep'].isin(train_rep)]
            valid = tested_sub_df.loc[tested_sub_df['rep'].isin(valid_rep)]
            train_set = Pandas_Dataset(train.groupby('sample_index'))
            valid_set = Pandas_Dataset(valid.groupby('sample_index'))

        elif train_type == 'Reversal':
            # Filter out test rep
            tested_sub_df = tested_sub_df.loc[~tested_sub_df['rep'].isin(test_rep)]
             # Refactor rep from 0 to 4 for domain classifier
            tested_sub_df['rep'] = pd.factorize(tested_sub_df['rep'])[0]
            # For each subject, only one exercise for each repetition is done. let's take a casual repetition as validation
            grouped= tested_sub_df.groupby('label')
            valid_sample_idx = []
            for lbl, group_df in grouped:
                # Get 20% of each exercise as validation
                num_samples = int(len(group_df['sample_index'].unique()) * 0.2) 
                print(f'Label: {lbl}    Valid samples: {num_samples}    Total:{len(group_df["sample_index"].unique())}')
                valid_sample_idx.append(group_df.sample(n=num_samples, random_state=29)['sample_index'].values)
            # Concatenate the arrays in valid_sample_idx_list vertically
            valid_sample_idx = np.concatenate(valid_sample_idx)
            # Retrieve the sampled rows from the original DataFrame using the sampled indices
            valid = tested_sub_df.loc[tested_sub_df['sample_index'].isin(valid_sample_idx)]
            train = tested_sub_df.loc[~tested_sub_df['sample_index'].isin(valid_sample_idx)]
            if (len(valid) + len(train) != len(tested_sub_df)):
                raise ValueError(' Train and validation splitting is not consistent')
            train_set = Pandas_Dataset(train.groupby('sample_index'), target_col='rep')
            valid_set = Pandas_Dataset(valid.groupby('sample_index'), target_col='rep')

            # Re-initialize the weights of domain classifier, previously used to classify subjects,
            # now will be use for different repetitions
            model.initialize_domain_class_weights()
            filename = filename +'_Reversal'

        # elif train_type == 'Triplet':           # 4 output
        #     train_set = dataframe_dataset_triplet(df_train, groupby_col='sample_index', target_col='sub')
        #     valid_set = dataframe_dataset_triplet(df_val, groupby_col='sample_index', target_col='sub')

        # elif train_type == 'JS':        # 3 output
        #     train_set = dataframe_dataset_JS(df_train, groupby_col='sample_index', target_col='sub')
        #     valid_set = dataframe_dataset_JS(df_val, groupby_col='sample_index', target_col='sub')

        # Correct only in the case last batch is exaclty 1 (it stops for batch_norm)
        train_dl = adjust_dataloader(train_set, **dataloader_param)
        valid_dl = adjust_dataloader(valid_set, **dataloader_param)

        if train_type == 'Standard':
            best_weights, tr_losses, val_losses = train_model_standard(model=model, loss_fun=loss_fn,
                                                                    optimizer=optimizer,
                                                                    dataloaders={"train": train_dl,
                                                                                    "val": valid_dl},
                                                                    num_epochs=num_epochs, precision=1e-5,
                                                                    scheduler=scheduler,
                                                                    patience=4, device=device,
                                                                    patience_increase=5)
        elif train_type == 'Reversal': 
            best_weights, tr_losses, val_losses, tr_dom_losses, val_dom_losses, tr_task_losses, val_task_losses = \
            train_model_reversal_gradient(model=model, loss_fun=loss_fn, loss_fun_domain=loss_fn,
                                      lamba=0.5, optimizer=optimizer, scheduler=scheduler,
                                      dataloaders={"train": train_dl, "val": valid_dl}, device=device,
                                      num_epochs=num_epochs , precision=1e-6, patience=8, patience_increase=7)
            

        # Save state dict of the model
        if not os.path.exists(path_to_save + 'Best_States/'):
            os.makedirs(path_to_save + 'Best_States/')
        torch.save(best_weights['state_dict'], path_to_save + f'Best_States/{filename}_Val_{valid_rep}.pth')


        # % PlotLoss and Confusion Matrix for subject
        if not os.path.exists(path_to_save + 'Plot/'):
            os.makedirs(path_to_save + 'Plot/')
        PlotLoss(tr_losses, val_loss=val_losses,
                 title=f'TL over Sub: {str(sub)} over {len(train_rep)} repetitions',
                 path_to_save=path_to_save + 'Plot/', filename=filename + '.png')
        
        if train_type == 'Reversal' or train_type == 'JS':
            PlotLoss(tr_dom_losses, val_loss=val_dom_losses,
            title=f'TL over Sub: {str(sub)} over {len(train_rep)} repetitions',
            path_to_save=path_to_save + '/Plot/',
            filename=f'{filename}_DOMAIN_ONLY.png')

            PlotLoss(tr_task_losses, val_loss=val_task_losses,
                title=f'TL over Sub: {str(sub)} over {len(train_rep)} repetitions',
                path_to_save=path_to_save + '/Plot/',
                filename=f'{filename}_TASK_ONLY.png')

    else:       # If I'm not training

        best_weights = {'epoch': 0} # need dict      to write in csv file
        val_losses = [0, 0]         # need iterable

    # Evaluate Model
    if not os.path.exists(path_to_save + 'Conf_Matrix/'):
        os.makedirs(path_to_save + 'Conf_Matrix/')
    softmax_block = torch.nn.Softmax(dim=1)
    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_dl:
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
        plot_confusion_matrix(cm, target_names=exe_labels, title=f'TL for subject {sub} over {len(train_rep)}_Reps',
                              path_to_save=path_to_save + 'Conf_Matrix/' + f'{filename}_Val_{valid_rep}.png')

    # csv file
    header_net = ['Name', 'Tested_sub', f'Train Repetitions', 'Val Repetition', 'Best Val Loss',
                  'Accuracy', 'Kappa', 'F1_score', 'Best Epoch', 'Num trainable Param']

    with open(path_to_save + f'Evals.csv', 'a', newline='') as myFile:
        writer = csv.writer(myFile)
        if myFile.tell() == 0:
            writer.writerow(header_net)
        # Create the row of values
        row = [filename, sub, train_rep, valid_rep, min(val_losses),
               metrics.balanced_accuracy_score(y_true=y_true, y_pred=y_pred),
               metrics.cohen_kappa_score(y1=y_true, y2=y_pred, weights=None),
               metrics.f1_score(y_true=y_true, y_pred=y_pred, average='weighted'),
               best_weights['epoch'], sum([p.numel() for p in model.parameters() if p.requires_grad])]
        writer.writerow(row)

    return print(f'Results saved in {path_to_save} !')


#%% Multi-head Attention

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels, height, width, group_size=64):
        super(PositionalEncoding2D, self).__init__()
        assert channels % group_size == 0, "Channels must be divisible by the group size"
        
        self.height = height
        self.width = width
        self.channels = channels
        self.group_size = group_size
        
        # Create positional encodings
        self.positional_encoding = self.create_positional_encoding(channels, height, width)

    def create_positional_encoding(self, channels, height, width):
        """
        Create a 2D positional encoding with unique channel encodings and positional values.
        """
        assert channels % 2 == 0  # Ensure channels are even for proper sine/cosine split

        pe = torch.zeros(channels, height, width)  # Positional encoding tensor

        # 1. **Channel-wise Encoding**: Apply a unique value for each feature map to encode the channel
        for group_start in range(0, channels, self.group_size):
            for i in range(group_start, group_start + self.group_size):
                pe[i, :, :] += i  # Add a unique value for each channel

        # 2. **2D Positional Encoding**: Apply sine and cosine positional encodings on the grid
        d_model = self.group_size // 2  # For sine and cosine encoding within each group

        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        
        pos_y = torch.arange(0, height).unsqueeze(1).repeat(1, width).float()
        pos_x = torch.arange(0, width).unsqueeze(0).repeat(height, 1).float()

        for i in range(0, d_model, 2):
            pe[i, :, :] += torch.sin(pos_y * div_term[i // 2])
            pe[i + 1, :, :] += torch.cos(pos_y * div_term[i // 2])

        for i in range(d_model, self.group_size, 2):
            pe[i, :, :] += torch.sin(pos_x * div_term[i // 2 - d_model // 2])
            pe[i + 1, :, :] += torch.cos(pos_x * div_term[i // 2 - d_model // 2])

        # 3. **Repeat Encoding Across Groups**: Ensure equal encoding for each group of 64 channels
        for group_start in range(self.group_size, channels, self.group_size):
            pe[group_start:group_start + self.group_size, :, :] = pe[0:self.group_size, :, :]

        return pe.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        # Add positional encoding to the input features
        return x + self.positional_encoding.to(x.device)


class MKCNN_grid_ATTN(nn.Module):
    def __init__(self, dict, num_domains, num_classes=14, num_heads=2):
        super(MKCNN_grid_ATTN, self).__init__()

        self.activation_fun = dict['act_func']
        self.in_ch = dict['N_multik'] * 2 * len(dict['Kernel_multi_dim'])  # Based on concatenation of MultiKernelConv2D

        # Multi-Kernel Conv2D grid layer (parallel convolutional feature extraction)
        self.parallel_convs = MultiKernelConv2D_grid(dict)

        # Positional encoding shared across all parallel convs
        H, W = 10, 7  # Assuming height and width are passed in dict
        self.positional_encoding = PositionalEncoding2D(channels=dict['N_multik'] * 2, height=H, width=W, group_size=dict['N_multik'] * 2)

        # Multi-Head Self-Attention layer
        self.multihead_attention = nn.MultiheadAttention(len(dict['Kernel_multi_dim']) * dict['N_multik'] * 2 * H * W, num_heads=num_heads)
                                                            # 5 * 32 * 2 * 10 * 7 -> 128, 10, 7
        # Final linear projection to reduce dimensions after attention
        # self.proj_linear = nn.Linear(len(dict['Kernel_multi_dim']) * dict['N_multik'] * 2, 4 * dict['N_multik'])
                                        

        self.conv_post_attentiont = nn.Sequential(
            SeparableConv2d(dict['N_Conv_conc'], dict['N_SepConv'], 1, kernel_size=3),  # Separable Conv 1
            nn.BatchNorm2d(dict['N_SepConv']),  # BatchNorm 1
            self.activation_fun,  # Activation
            nn.MaxPool2d(2),  # Pool layer (window size adjusted as per reduced dimensions)
            nn.Dropout2d(0.2),  # Dropout

            SeparableConv2d(dict['N_SepConv'], dict['N_SepConv'], 1, kernel_size=3),  # Separable Conv 2
            nn.BatchNorm2d(dict['N_SepConv']),  # BatchNorm 2
            self.activation_fun,  # Activation
            nn.MaxPool2d(2),  # Pooling after separable conv
            nn.Dropout2d(0.2),  # Dropout

            nn.Flatten(),  # Flatten before fully connected layers
        )


       # Fully connected layers for final classification
        self.fc_layers = nn.Sequential(
            nn.Linear(2 * dict['N_SepConv'], 128),  # Linear with reduced size
            nn.BatchNorm1d(128),
            self.activation_fun,  # Activation function
            nn.Linear(128, num_classes)  # Final classification layer
        )

        # Domain classifier for adversarial training
        self.domain_classifier = nn.Sequential(
            nn.Linear(2 * dict['N_SepConv'], 128),  # First layer
            nn.BatchNorm1d(128),  # Batch norm
            self.activation_fun,  # Activation
            nn.Linear(128, num_domains)  # Output layer for domain classification
        )
        
    def forward(self, x, alpha=None):
        # Pass input through the parallel convolutions (MultiKernelConv2D_grid)
        features = self.parallel_convs(x)  # Shape: [batch, 320, H, W]

        # Apply positional encoding to each feature map
        features = torch.split(features, 64, dim=1)  # Split feature maps into 5 groups of 64
        encoded_features = [self.positional_encoding(fmap) for fmap in features]
        for i, enc_fmap in enumerate(encoded_features):
            print(f"Feature map {i} shape after positional encoding: {enc_fmap.shape}")

        # Concatenate the encoded features
        concat_features = torch.cat(encoded_features, dim=1)  # Shape: [batch, 320, H, W]
        print(f"Shape before attention: {concat_features.shape}")

        # Flatten spatial dimensions and apply attention
        batch_size, channels, H, W = concat_features.shape
        flattened_features = concat_features.view(batch_size, channels, H * W)
        attention_output, _ = self.multihead_attention(flattened_features, flattened_features, flattened_features)

        # Reshape back to original structure
        reshaped_output = attention_output.view(batch_size, 64 * 5, H, W)

        # Pass through post-attention convolutional layers
        post_attention_output = self.conv_post_attentiont(reshaped_output)
        
        class_output = self.fc_layers(post_attention_output)

        if alpha is not None:
            # Apply Gradient Reversal Layer
            reversed_features = GradientReversalLayer.apply(post_attention_output, alpha)

            # Domain classification output
            domain_output = self.domain_classifier(reversed_features)
            return class_output, domain_output

        # If no adversarial training is happening, only return the classification output
        return class_output

#%% COPIED FOR ATTENTION
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale
def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale
class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=None, no_spatial=False):
        super(CBAM, self).__init__()
        if pool_types is None:
            pool_types = ['avg', 'max']
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

import re
import time
import os
import sys
import random
import copy
import csv

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from scipy.stats import loguniform
from sklearn import metrics

from DATALOADERS import JSDLoss, CenterLoss
from Data_Processing_Utils import PlotLoss, plot_confusion_matrix
from MODELS import MKCNN_grid


# %% Training types
def train_model_standard(model, loss_fun, optimizer, dataloaders, scheduler, num_epochs=150, precision=1e-8,
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

                    outputs, _ = model.forward(inputs)
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
                    # print(70 * '#', 5 * '\n',f'\nPATIENCE: -> {patience}',5 * '\n', 70 * '#')
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - epoch_start), 5 * '\n')
        if epoch > patience:
            break

    tr_epoch_loss_vec, val_epoch_loss_vec = np.array(tr_epoch_loss_vec), np.array(val_epoch_loss_vec)
    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    return best_state, tr_epoch_loss_vec, val_epoch_loss_vec

#%% Pre_train_Triplet
def pre_train_model_triplet(model, loss_fun, optimizer, dataloaders, scheduler, num_epochs=150, precision=1e-8,
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
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - epoch_start), 5 * '\n')
        if epoch > patience:
            break

    tr_epoch_loss_vec, val_epoch_loss_vec = np.array(tr_epoch_loss_vec), np.array(val_epoch_loss_vec)
    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    return best_state, tr_epoch_loss_vec, val_epoch_loss_vec

# Train_Triplet
def train_model_triplet(model, loss_fun, optimizer, dataloaders, scheduler, num_epochs=100, precision=1e-8,
                        patience=10, patience_increase=10, device=None, margin=1, p_dist=2, beta=1.0):
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
    triplet_loss = nn.TripletMarginLoss(margin=margin, p=p_dist)

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
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - epoch_start), 5 * '\n')
        if epoch > patience:
            break

    tr_epoch_loss_vec, val_epoch_loss_vec = np.array(tr_epoch_loss_vec), np.array(val_epoch_loss_vec)
    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    return best_state, tr_epoch_loss_vec, val_epoch_loss_vec

#%% Train model reversal
def train_model_reversal_gradient(model, loss_fun, optimizer, dataloaders, scheduler, lamba = 0.5, alpha_start = 0, 
                                  num_epochs = 100, precision = 1e-8, patience = 10, patience_increase = 10, 
                                  loss_fun_domain=nn.CrossEntropyLoss(), combined_val_loss =  False, check_val_dom_loss = True, device=None):
    '''
    Parameters:
        model (torch.nn.Module):                           The neural network model to be trained.
        loss_fun (torch.nn.modules.loss):                  The loss function used for calculating the task-specific loss.
        optimizer (torch.optim.Optimizer):                 The optimizer used for updating the model's weights.
        dataloaders (dict):                                A dictionary containing torch.utils.data.DataLoader instances for both training ('train') and validation ('val') phases.
        scheduler (torch.optim.lr_scheduler):              Scheduler to adjust the learning rate based on the training progress.
        lamba (float, optional, default=0.5):              Weighting factor for the domain adaptation loss in the total loss calculation.
        alpha_start (int, optional, default=0):            The starting epoch from which the domain adaptation effect begins.
        num_epochs (int, optional, default=100):           Total number of epochs for training the model.
        precision (float, optional, default=1e-8):         A threshold for numerical precision in comparing floating-point numbers.
        patience (int, optional, default=10):              The number of epochs to wait for improvement in validation loss before early stopping.
        patience_increase (int, optional, default=10):     The increase in patience when a new best validation loss is found.
        loss_fun_domain (torch.nn.modules.loss, optional): The loss function used for the domain adaptation task.
        combined_val_loss (bool, optional, default=False): Whether to combine task-specific and domain adaptation losses during validation.
        check_val_dom_loss (bool, optional, default=True): Whether to compute and check the domain loss during the validation phase.
        device (torch.device, optional):                   The device (CPU or GPU) on which to perform computations. If not specified, it defaults to GPU if available, otherwise CPU.
    Returns:
        best_state (dict):                     A dictionary containing the best model state, optimizer state, and scheduler state based on validation loss.
        tr_epoch_loss (numpy.ndarray):         Array of training losses for each epoch.
        val_epoch_loss (numpy.ndarray):        Array of validation losses for each epoch.
        tr_epoch_loss_domain (numpy.ndarray):  Array of training domain losses for each epoch.
        val_epoch_loss_domain (numpy.ndarray): Array of validation domain losses for each epoch.
        tr_epoch_loss_task (numpy.ndarray):    Array of training task-specific losses for each epoch.
        val_epoch_loss_task (numpy.ndarray):   Array of validation task-specific losses for each epoch.
    '''
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
    # alpha_vec = []

    if combined_val_loss:
        print(70*'#', 5*'\n','Considering domain loss over total loss on validation!', 5*'\n', 70*'#')
        
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

            for inputs, labels, sub in dataloaders[phase]:
                cc = cc + 1

                inputs = inputs[:, None, :, :]  # Adding on
                inputs = inputs.to(device)

                labels = labels.type(torch.LongTensor).to(device)  # Loss_Fun want LongTensor
                labels_np = labels.cpu().data.numpy()

                sub = sub.type(torch.LongTensor).to(device)
                # print(f'LABEL: {labels[0]}      SUB:{sub[0]}')
                if phase == 'train':
                    
                    # Increasing domain loss 
                    # p = float(epoch * tot_batch) / num_epochs / tot_batch

                    # if epoch >= alpha_start:
                    #     alpha = 2. / (1. + np.exp(-10 * p)) - 1
                    #     # alpha = 1
                    # else:
                    #     alpha = 0
                        
                    # Decreasing or increasing domain loss with decays
                    alpha = get_exponential_function(epoch, num_epochs, alpha_start=0.5, alpha_lim=1.5, fraction=0.3)
                    

                    model.train()

                    outputs, out_domain = model.forward(inputs, alpha=alpha)
                    # print(f'O1: {outputs.shape}     O2: {out_domain.shape}')
                    outputs_softmax = softmax_block(outputs)

                    outputs_np = outputs_softmax.cpu().data.numpy()
                    outputs_np = np.argmax(outputs_np, axis=1)

                    y_true = np.append(y_true, outputs_np)
                    y_pred = np.append(y_pred, labels_np)

                    loss_task = loss_fun(outputs, labels)
                    loss_domain = loss_fun_domain(out_domain, sub)
                    
                    # loss = loss_task + lamba * (1/(1 + loss_domain))
                    # Use a weighted average 
                    loss = (1 - lamba) * loss_task + lamba * (1 / (1 + loss_domain))

                    loss.backward()
                    optimizer.step()

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    model.zero_grad()


                else:
                    model.eval()
                    with torch.no_grad():
                        # forward
                        outputs, out_domain = model(inputs, alpha=alpha)

                        outputs_softmax = softmax_block(outputs)
                        outputs_np = outputs_softmax.cpu().data.numpy()
                        outputs_np = np.argmax(outputs_np, axis=1)

                        y_true = np.append(y_true, outputs_np)
                        y_pred = np.append(y_pred, labels_np)

                        loss_task = loss_fun(outputs, labels)
                        if check_val_dom_loss:
                            loss_domain = loss_fun(out_domain, sub)
                        
                        if combined_val_loss:
                            # This limit is set in order to avoid the model to find the point with lower valid loss
                            # over the domain, infact, at the very beginning, with alpha zero, the invariant feature are not
                            # learned, so the model will have a lower TOTAL validation loss
                            # if alpha > 0.8:   # alpha > 0.8 -->  with 100 epochs and alpha_start = 0 it starts to count domain loss at epoch 22
                            # loss = loss_task + lamba * (1/(1 + loss_domain))
                            loss = (1 - lamba) * loss_task + lamba * (1 / (1 + loss_domain))


                        else:
                            loss = loss_task
                        
                        
                task_loss_contribution = ((1 - lamba) * loss_task.item()) / loss.item()
                domain_loss_contribution = ((lamba * (1 / (1 + loss_domain.item()))) / loss.item()) * alpha

                loss_accumulator = loss_accumulator + ((1 / (total + 1)) * (loss.item() - loss_accumulator))
                domain_loss_acc = domain_loss_acc + ((1 / (total + 1)) * (loss_domain.item() - domain_loss_acc))
                task_loss_acc = task_loss_acc + ((1 / (total + 1)) * (loss_task.item() - task_loss_acc))
                total = total + 1

                acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
                # kappa = metrics.cohen_kappa_score(y1=y_true, y2=y_pred, weights='quadratic')
                print(phase + " -> loss: {:0.5f}".format(loss_accumulator) + ", accuracy: {:0.5f}".format(acc) + \
                    ", task_loss_contrib: {:0.4f}".format(task_loss_contribution) + 
                    ", dom_loss_contrib: {:0.4f}".format(domain_loss_contribution) + \
                        ", task_loss: {:0.4f}".format(task_loss_acc) + ", dom_loss: {:0.4f}".format(domain_loss_acc) + 
                    f', Epoch: -> {epoch} / {num_epochs}  Batch Number: -> {cc} / {tot_batch}')

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
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - epoch_start), 5 * '\n')
        if epoch > patience:
            break
            
        print(70 * '#', 5 * '\n', f'ALPHA: -> {alpha}', 5 * '\n', 70 * '#')
        # Inserted to check values
        # alpha_vec.append(alpha)
        # print(70 * '#', 5 * '\n', f'ALPHA_VEC: -> {alpha_vec}', 5 * '\n', 70 * '#')

    tr_epoch_loss, val_epoch_loss = np.array(tr_epoch_loss), np.array(val_epoch_loss)
    tr_epoch_loss_domain, val_epoch_loss_domain = np.array(tr_epoch_loss_domain), np.array(val_epoch_loss_domain)
    tr_epoch_loss_task, val_epoch_loss_task = np.array(tr_epoch_loss_task), np.array(val_epoch_loss_task)
    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    return best_state, tr_epoch_loss, val_epoch_loss, tr_epoch_loss_domain, val_epoch_loss_domain, tr_epoch_loss_task, val_epoch_loss_task

#%% Train model JS
def train_model_JS(model, loss_fun, optimizer, dataloaders, scheduler, lambda1=0.5, lambda2=0.5,
                        num_epochs=100, precision=1e-8,
                        patience=10, patience_increase=10, device=None):
    if not device:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device("cpu")

    # model.load_state_dcit(torch.load(source_domain_path))
    # source_embedding = nn.Embedding(14, model[-1].in_features)

    # DAM
    criterion_JSD = JSDLoss().to(device)

    # DNM - > set num_classes and feat_dim according to write parameters
    criterion_center_loss = CenterLoss(14, 128, device=device)
    criterion_center_loss.centers.to(device)

    # cross entropy
    criterion = torch.nn.CrossEntropyLoss()

    since = time.time()
    tr_epoch_loss = []
    val_epoch_loss = []
    tr_epoch_loss_JS = []
    val_epoch_loss_JS = []
    tr_epoch_loss_center = []
    val_epoch_loss_center = []
    tr_epoch_loss_task = []
    val_epoch_loss_task = []

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
            print(70 * '#', 5 * '\n', f'PHASE: -> {phase}', 5 * '\n', 70 * '#')
            tot_batch = len(dataloaders[phase])
            total = 0
            cc = 0
            loss_accumulator = 0.0
            JS_loss_acc = 0.0
            center_loss_acc = 0.0
            task_loss_acc = 0.0

            y_true = []
            y_pred = []

            for inputs, labels, pos_sample in dataloaders[phase]:
                cc = cc + 1

                inputs, pos_sample = inputs[:, None, :, :], pos_sample[:, None, :, :]  # Adding on
                inputs, pos_sample = inputs.to(device), pos_sample.to(device)

                labels = labels.type(torch.LongTensor).to(device)  # Loss_Fun want LongTensor
                labels_np = labels.cpu().data.numpy()


                if phase == 'train':
                    model.train()

                    outputs, features = model(inputs)
                    _, out_pos = model(pos_sample)

                    # print(f'O1: {outputs.shape}     O2: {out_domain.shape}')
                    outputs_softmax = softmax_block(outputs)

                    outputs_np = outputs_softmax.cpu().data.numpy()
                    outputs_np = np.argmax(outputs_np, axis=1)

                    y_true = np.append(y_true, outputs_np)
                    y_pred = np.append(y_pred, labels_np)

                    loss_task = loss_fun(outputs, labels)
                    features = features.to(device)
                    loss_domain = criterion_JSD(features, out_pos)
                    loss_center = criterion_center_loss(features, labels)
                    criterion_center_loss.update_centers(features, labels)

                    loss = loss_task + (lambda1 * loss_domain) + lambda2 * loss_center
                    loss.backward()
                    optimizer.step()

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    model.zero_grad()


                else:
                    model.eval()
                    with torch.no_grad():
                        # forward
                        outputs, out_domain = model(inputs)

                        outputs_softmax = softmax_block(outputs)
                        outputs_np = outputs_softmax.cpu().data.numpy()
                        outputs_np = np.argmax(outputs_np, axis=1)

                        y_true = np.append(y_true, outputs_np)
                        y_pred = np.append(y_pred, labels_np)

                        loss_task = loss_fun(outputs, labels)
                        loss_domain = criterion_JSD(features, out_pos)
                        loss_center = criterion_center_loss(features, labels)
                        loss = loss_task + (lambda1 * loss_domain) + lambda2 * loss_center


                loss_accumulator = loss_accumulator + ((1 / (total + 1)) * (loss.item() - loss_accumulator))
                JS_loss_acc = JS_loss_acc + ((1 / (total + 1)) * (loss_domain.item() - JS_loss_acc))
                center_loss_acc = center_loss_acc + ((1 / (total + 1)) * (loss_center.item() - center_loss_acc))
                task_loss_acc = task_loss_acc + ((1 / (total + 1)) * (loss_task.item() - task_loss_acc))
                total = total + 1

                acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
                kappa = metrics.cohen_kappa_score(y1=y_true, y2=y_pred, weights='quadratic')
                print(phase + " -> loss: {:0.5f}".format(loss_accumulator) + ", accuracy: {:0.4f}".format(acc) +
                      ", kappa: {:0.4f}".format(kappa) + ', JS_loss: {:0.4f}'.format(JS_loss_acc) + ', Centre_L: {:0.4f}'.format(center_loss_acc)
                      + f', Epoch: -> {epoch}/{num_epochs}  Batch: -> {cc}/{tot_batch}')

            if phase == 'train':
                tr_epoch_loss.append(loss_accumulator)
                tr_epoch_loss_JS.append(JS_loss_acc)
                tr_epoch_loss_center.append(center_loss_acc)
                tr_epoch_loss_task.append(task_loss_acc)
            elif phase == 'val':
                val_epoch_loss.append(loss_accumulator)
                val_epoch_loss_JS.append(JS_loss_acc)
                val_epoch_loss_center.append(center_loss_acc)
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
    tr_epoch_loss_JS, val_epoch_loss_JS = np.array(tr_epoch_loss_JS), np.array(val_epoch_loss_JS)
    tr_epoch_loss_center, val_epoch_loss_center = np.array(tr_epoch_loss_center), np.array(val_epoch_loss_center)
    tr_epoch_loss_task, val_epoch_loss_task = np.array(tr_epoch_loss_task), np.array(val_epoch_loss_task)
    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    return best_state, \
        tr_epoch_loss, val_epoch_loss, \
        tr_epoch_loss_JS, val_epoch_loss_JS, \
        tr_epoch_loss_center, val_epoch_loss_center, \
        tr_epoch_loss_task, val_epoch_loss_task


# %% Random grid
def random_choice(dictionary):
    '''

    Selects at random a set of parameters from a given dictionary 'dict'

    '''
    par = {}
    for key, value in dictionary.items():  # If you put two variable inside this iterable (items) it automatically gets
        par[key] = random.choice(value)  # as first the keys (tuple) and as second the values inside that key.
    return par


def random_GS(grid, train_set, valid_set, test_set, path_to_save, cm_labels=None):
    """
    :param grid:            Dictionary nesting two other dictionary: 'net' & 'learning'. The first one will give infos
                            about how a parametric model is going to be build (e.g. kernels dim, N neurons, act. fun...);
                            the second one will give information about model specific hyper params (e.g. epochs, N model to test, lr...)

    :param train_set:       As for valid and test, is the dataset which incorporates at least the basics of a pytorch
                            dataloader (_get_item_ & _len_) such that can be build a torch.utils.data.Dataloader(train_set)

    :param cm_labels:       Label for Confusion_Matrix
    :param path_to_save:    Where to save models parameters, Plots & Confusion Matrix: given a folder it will create 3
                            subfolders named: /State_Dict, /Plot & /Conf_Matrix containing respectively the weights &
                            structure of the model, the images related to the Loss Plot over epochs and the Conf_Matrix;
                            It will also create a Params.csv file that will contain all the model necessary information.
    :param n_model_to_keep: Number of the model to save: If not provided, it will save all the models infos, if provided,
                            it will only save the best n_models based on val_loss (or test_loss if test_dataset).
                            Consider that to give in input n_model_to_keep will make the system keeping in memory all
                            the infos about n_models for all the time random_GS is running.
    :return:                It creates 3 directories nested in path_to_save variable which contain Plots, Confusion Matrixes &
                            Ordered_state_dict of the models. It also creates a csv file called Params which resume
                            models infos.
    """

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    if not os.path.exists(path_to_save + '/Plot/'):
        os.makedirs(path_to_save + '/Plot/')
    if not os.path.exists(path_to_save + '/Conf_Matrix/'):
        os.makedirs(path_to_save + '/Conf_Matrix/')
    if not os.path.exists(path_to_save + '/State_Dict/'):
        os.makedirs(path_to_save + '/State_Dict/')

    # Resuming grid in the case of stopping
    if os.path.exists(path_to_save + '/Params.csv'):
        # Read the last saved model number from the CSV file
        with open(path_to_save + '/Params.csv', 'r') as file:
            reader = csv.reader(file)
            last_row = list(reader)[-1]  # Get the last row
            last_model_number = int(last_row[0])  # Get the first column of the last row
            start_model_number = last_model_number + 1
        print(f"Resuming from model number: {start_model_number}")
    else:
        start_model_number = 0

    h_n = grid['net']  # selecting network hyperparameter
    h_l = grid['learning']  # selecting simulation hyperparameter

    # loop over n models
    for ii in range(start_model_number,
                    h_l['n_models'][0]):  # 'n_models' is a list object, so I MUST indicate position [0]

        print(70 * '#', 5 * '\n', f'Trying model number {ii + 1} /', h_l['n_models'][0])

        # choose parameters at random from the grid
        n_params = random_choice(h_n)
        l_params = random_choice(h_l)

        # total dictionary to append at the end
        total_params = {'net': n_params, 'learning': l_params}
        print(total_params)

        # define data loaders
        train_dataloader = data.DataLoader(train_set, batch_size=int(l_params['batch_size']), shuffle=True,
                                           num_workers=int(l_params['num_workers']))
        valid_dataloader = data.DataLoader(valid_set, batch_size=int(l_params['batch_size']), shuffle=True,
                                           num_workers=int(l_params['num_workers']))
        dataloaders = {"train": train_dataloader, "val": valid_dataloader}

        # Build model
        net = MKCNN_grid(n_params, num_classes=14).to(l_params['device'])
        net.reset()
        # optimizer selection
        if l_params['opt'] == 'Adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=l_params['lr'], betas=(0.95, 0.9), amsgrad=True)
        else:
            optimizer = torch.optim.Adam(net.parameters(), lr=l_params['lr'], betas=(0.95, 0.9), amsgrad=True)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=.2,
                                                               patience=5, verbose=True, eps=1e-6)


        best_state, tr_loss, val_loss = train_model_standard(net, dataloaders=dataloaders, loss_fun=l_params['loss_fn'],
                                                             optimizer=optimizer, scheduler=scheduler,
                                                             num_epochs=l_params['num_epochs'])

        # Saving state dict
        torch.save(best_state['state_dict'], path_to_save + f'/State_Dict/state_dict_model_{ii + 1}.pth')
        # Saving Train/Val Plot
        PlotLoss(tr_loss, val_loss=val_loss, title='Model Number ' + str(ii + 1),
                 path_to_save=path_to_save + '/Plot/', filename=f'model_N{str(ii + 1)}.png')
        # Num params
        num_params = sum(p.numel() for p in net.parameters())

        # Saving Confusion Matrix
        y_true = []
        y_pred = []
        test_dataloader = data.DataLoader(test_set, batch_size=int(l_params['batch_size']), shuffle=True)
        net.load_state_dict(best_state['state_dict'])  # Loading weights in the net of current model
        net.eval()
        softmax_block = nn.Softmax(dim=1)
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                # inputs = torch.swapaxes(inputs, 2, 1)  # -> convert from [10,20] to [20,10]
                inputs = inputs[:, None, :, :]
                inputs = inputs.to(l_params['device'])
                # labels = labels.type(torch.FloatTensor).to(l_parmas['device'])
                # labels = labels.to(l_parmas['device'])
                labels_np = labels.cpu().data.numpy()
                # print(labels_np.shape)
                # forward
                outputs, _ = net(inputs)
                outputs_softmax = softmax_block(outputs)
                outputs_np = outputs_softmax.cpu().data.numpy()
                outputs_np = np.argmax(outputs_np, axis=1)
                # print(outputs_np.shape)

                y_pred = np.append(y_pred, outputs_np)
                y_true = np.append(y_true, labels_np)

        cm = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
        if cm_labels is None:
            cm_labels = np.arange(len(cm)) + 1
        plot_confusion_matrix(cm, target_names=cm_labels, title=f'Confusion Matrix Model number {ii + 1}',
                              path_to_save=path_to_save + f'/Conf_Matrix/model_N#{ii + 1}')

        # Average inference time model-based to save
        time_dataloader = data.DataLoader(test_set, batch_size=1, shuffle=True)
        iterable_dataloader = iter(time_dataloader)
        with torch.no_grad():
            time_tot = 0
            for t in range(100):
                j, _ = next(iterable_dataloader)
                j = j[:, None, :, :].to(l_params['device'])
                start_time = time.time()
                _ = net(j)
                time_tot = time_tot + (time.time() - start_time)
            avg_time = (time_tot / 100) * 1000

        # Preparing the row
        row = [ii + 1, min(val_loss),
               metrics.accuracy_score(y_true=y_true, y_pred=y_pred),
               metrics.cohen_kappa_score(y1=y_true, y2=y_pred, weights='quadratic'),
               metrics.f1_score(y_true=y_true, y_pred=y_pred, average='weighted'),
               best_state['epoch'], avg_time, num_params, best_state['scheduler']]

        # Write CSV file with relevant infos about models
        with open(path_to_save + '/Params.csv', 'a', newline='') as myFile:
            writer = csv.writer(myFile)  # -> writer Object
            if ii == 0:  # Building columns (header net) only first cycle
                header_net = ['N_model', 'Best Val Loss', 'Accuracy', 'Kappa', 'F1_score', 'Best Epoch',
                              'Inference time [ms]', 'Num Params', 'scheduler']
                for key in total_params.keys():  # Key
                    for par in total_params[key].keys():  # Append columns of both net_params and learning_params
                        header_net.append(par)
                header_net = header_net[:-2]
                writer.writerow(header_net)

            for param_type in total_params.keys():  # -> param_type refers to -> net or learning
                for key in total_params[param_type].keys():
                    row.append(str(total_params[param_type][key]))
            row = row[:-2]
            writer.writerow(row)

    return print(f'Results saved in {path_to_save} & SubFolders', 70 * '#', 5 * '\n')


def get_l2_regularization(model, lambda_l2):
    l2_reg = 0.0
    for param in model.parameters():
        l2_reg += torch.norm(param, p=2).pow(2)
    l2_reg *= lambda_l2
    return l2_reg

import numpy as np

import numpy as np

#%%
import numpy as np

def get_exponential_function(epoch, num_epochs, alpha_start=1.0, alpha_lim=0.25, fraction=2/3):
    """
    Calculate an exponentially decaying or increasing alpha value based on the comparison of alpha_start and alpha_lim.
    If alpha_start > alpha_lim, it decays from alpha_start to alpha_lim. If alpha_start < alpha_lim, it increases
    from alpha_start to alpha_lim after a specified fraction of total epochs.
    
    Parameters:
    - epoch: The current epoch (starting from 0).
    - num_epochs: The total number of epochs.
    - alpha_start: The initial value of alpha at the start of training.
    - alpha_lim: The target value of alpha at the end of decay or increase.
    - fraction: The fraction of total epochs by which alpha should reach alpha_lim (default is 2/3).
    
    Returns:
    - alpha: The calculated alpha value for the current epoch.
    """
    
    # Calculate the number of epochs over which the transition should occur
    transition_epochs = fraction * num_epochs
    
    # Determine if we are decaying or increasing based on the comparison of alpha_start and alpha_lim
    if alpha_start > alpha_lim:
        # Decaying from alpha_start to alpha_lim
        decay_rate = np.log(alpha_start / alpha_lim) / transition_epochs
        alpha = alpha_start * np.exp(-decay_rate * epoch)
        
        
        # Ensure alpha doesn't drop below alpha_lim
        alpha = max(alpha, alpha_lim)
    
    else:
        # Increasing from alpha_start to alpha_lim
        increase_rate = np.log(alpha_lim / alpha_start) / transition_epochs
        # alpha = alpha_start + (alpha_lim - alpha_start) * (1 - np.exp(-increase_rate * epoch))
        alpha = alpha_start * np.exp(increase_rate * epoch)
        
        # Ensure alpha doesn't exceed alpha_lim
        alpha = min(alpha, alpha_lim)
    
    return alpha

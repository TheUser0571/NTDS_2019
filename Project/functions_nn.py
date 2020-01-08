# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 15:51:22 2019

@author: kay-1
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging



def get_nn_inputs(x):
    """
    Creates the input tensor for the neural net
    """
    y = torch.zeros(int(x.shape[0] / 12), x.shape[1], 12)
    for i in range(int(x.shape[0] / 12)):
        y[i] = torch.from_numpy(x[i*12:(i+1)*12, :].T)
    return y

def retrieve_ts_from_nn_outputs(x):
    """
    Retrieves the two dimensional numpy time series from the three dimensional tensor format
    """
    x = x.squeeze()
    y = np.zeros((x.shape[0]*x.shape[2], x.shape[1]))
    for i in range(x.shape[0]):
        y[i*x.shape[2]:(i+1)*x.shape[2], :] = x[i].detach().numpy().T
    return y

# -----------------------------------------------------------------------------------------------------------------

def standardize(x):
    """
    Standardizes the torch data to mean=0 and std=1
    """
    std_data = torch.zeros(x.shape)
    if len(x.shape) > 2:
        mean = torch.mean(x.reshape((x.shape[0]*x.shape[1], x.shape[2])), axis=0)
        std = torch.std(x.reshape((x.shape[0]*x.shape[1], x.shape[2])), axis=0)
        for i in range(x.shape[0]):
            centered_data = x[i] - mean
            std_data[i] = centered_data / std
    else:
        mean = torch.mean(x, axis=0)
        std = torch.std(x, axis=0)
        centered_data = x - mean
        std_data = centered_data / std
    return std_data, mean, std

# -----------------------------------------------------------------------------------------------------------------

def fit_standardize(x, mean_stand, std_stand):
    """
    Standardizes the torch data to given mean and std
    """
    std_data = torch.zeros(x.shape)
    if len(x.shape) > 2:
        for i in range(x.shape[0]):
            std_data[i] = (x[i] - mean_stand) / std_stand
    else:
        std_data = (x - mean_stand) / std_stand
    return std_data

# -----------------------------------------------------------------------------------------------------------------

def train_test_set(feature, label, ratio=0.8):
    """
    Splits the data set into a train and test set according to the 'ratio'
    """
    train_sep = int(feature.shape[0] * ratio)
    train_input = feature[:train_sep]
    train_target = label[:train_sep]

    test_input = feature[train_sep:]
    test_target = label[train_sep:]

    return train_input, train_target, test_input, test_target

# -----------------------------------------------------------------------------------------------------------------

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = torch.randperm(data_size)
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            
            
def train(model, train_inputs, train_targets, test_inputs, test_targets, n_epoch=50, batch_size=10):
    model.reset_parameters()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.05)
    
    train_losses = []
    test_losses = []

    logging.debug('Training Start')
    for epoch in range(n_epoch):
        running_loss = 0
        for batch_labels, batch_inputs in batch_iter(train_targets, train_inputs, batch_size=batch_size,
                                                     num_batches=int(train_inputs.shape[0] / batch_size)):
            # zero the parameter gradients
            optimizer.zero_grad()
            # calculate outputs
            outputs = model(batch_inputs)
            # calculate loss
            train_loss = loss_fn(outputs, batch_labels)
            # perform backpropagation
            train_loss.backward()
            # optimize
            optimizer.step()
    
            running_loss += train_loss.item()
    
        epoch_loss = running_loss / int(train_inputs.shape[0] / batch_size)
        
        running_loss = 0
        for batch_labels, batch_inputs in batch_iter(test_targets, test_inputs, batch_size=batch_size,
                                                     num_batches=int(test_inputs.shape[0] / batch_size)):
            with torch.no_grad():
                pred = model(test_inputs)
                test_loss = loss_fn(pred, test_targets)
            running_loss += test_loss.item() / int(test_inputs.shape[0] / batch_size)
        
        train_losses.append(epoch_loss)
        test_losses.append(running_loss)
    
        # logging.debug statistics
        logging.debug(f'epoch: {epoch}, train loss: {epoch_loss}, test loss: {test_loss.item()}')
    logging.debug('Training finished!')
    ## ------------------------------------------------------------------------------------
    ## Saving the trained model to file----------------------------------------------------
    PATH = f'NeuralNet_trained.pt'
    logging.debug('Saving trained model to ' + PATH + '... ')
    torch.save(model, PATH)
    logging.debug('Saving trained model to ' + PATH + '... Done!')
    
    ## save losses
    logging.debug('Saving losses... ')
    np.savetxt(f'train_loss.txt', train_losses)
    np.savetxt(f'test_loss.txt', test_losses)
    logging.debug('Saving losses... Done!')
    
    return train_losses, test_losses
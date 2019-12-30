# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 15:26:47 2019

@author: kay-1
"""

import numpy as np
from functions_nn import *
import torch
from net import *
import pandas as pd

#BASE_PATH = 'C:/Users/kay-1/Documents/NTDS_data/'
BASE_PATH = ''
# %% load forecast data from single csv file (run this instead of the above if the csv file is already generated!)
print('Loading forecast data...')
solar_fc = np.loadtxt(BASE_PATH + 'solar_fc.csv', delimiter=',')
wind_fc = np.loadtxt(BASE_PATH + 'wind_fc.csv', delimiter=',')
print('Loading forecast data... Done!')
# %% load power capacities
print('Loading power capacities...')
# directly convert to numpy and extract the proportional capacities
solar_cp = pd.read_csv(BASE_PATH + 'solar_layouts_COSMO.csv').to_numpy()[:,1]
wind_cp = pd.read_csv(BASE_PATH + 'wind_layouts_COSMO.csv').to_numpy()[:,1]
print('Loading power capacities... Done!')
# %% load actual data
print('Loading actual data...')
solar_ts_complete = pd.read_csv(BASE_PATH + 'solar_signal_COSMO.csv').to_numpy()
solar_ts = solar_ts_complete[:,1:].astype(float)
wind_ts = pd.read_csv(BASE_PATH + 'wind_signal_COSMO.csv').to_numpy()[:,1:].astype(float)  # directly convert into numpy and remove the time column
print('Loading actual data... Done!')
# %% convert signals to MWh
print('Converting signals to MWh...')
solar_fc_MWh = solar_fc * solar_cp
solar_ts_MWh = solar_ts * solar_cp
wind_fc_MWh = wind_fc * wind_cp
wind_ts_MWh = wind_ts * wind_cp
print('Converting signals to MWh... Done!')
# %% machine learing
# initialize model

#net = NeuralNet(in_size=12, out_size=12)  # simple net
net = ConvNet()  # conv net

# %% prepare training and test data
print('Preparing training and test data...')
solar_fc_tensor = get_nn_inputs(solar_fc_MWh)
solar_ts_tensor = get_nn_inputs(solar_ts_MWh)
wind_fc_tensor = get_nn_inputs(wind_fc_MWh)
wind_ts_tensor = get_nn_inputs(wind_ts_MWh)

solar_train_feat, solar_train_target, solar_test_feat, solar_test_target = train_test_set(solar_fc_tensor, solar_ts_tensor)
wind_train_feat, wind_train_target, wind_test_feat, wind_test_target = train_test_set(wind_fc_tensor, wind_ts_tensor)

solar_train_feat_std, mean_solar, std_solar = standardize(solar_train_feat)
solar_test_feat_std = fit_standardize(solar_test_feat, mean_solar, std_solar)
wind_train_feat_std, mean_wind, std_wind = standardize(wind_train_feat)
wind_test_feat_std = fit_standardize(wind_test_feat, mean_wind, std_wind)
print('Preparing training and test data... Done!')
# %% for conv model
solar_train_feat_std = solar_train_feat_std.unsqueeze(1)
solar_test_feat_std = solar_test_feat_std.unsqueeze(1)

wind_train_feat_std = wind_train_feat_std.unsqueeze(1)
wind_test_feat_std = wind_test_feat_std.unsqueeze(1)

# %% train the model for solar engergy
train_loss, test_loss = train(model=net, train_inputs=wind_train_feat_std, train_targets=wind_train_target, 
                              test_inputs=wind_test_feat_std, test_targets=wind_test_target, n_epoch=50, batch_size=10)
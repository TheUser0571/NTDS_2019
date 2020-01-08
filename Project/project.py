# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import functions
from functions import *
import networkx as nx
import matplotlib
import matplotlib.image as mpimg
import torch
from net import *
import geopy.distance # conda install -c conda-forge geopy

# %%
BASE_PATH = 'P:/Python/Network_tour/Data_for_project/'  # adjust if necessary

# BASE_PATH = 'C:/Users/kay-1/Documents/NTDS_data/'  # Kay's path ;)

edges = pd.read_csv(BASE_PATH + 'edges.csv')
nodes = pd.read_csv(BASE_PATH + 'nodes.csv')
generator = pd.read_csv(BASE_PATH + 'generator.csv')
load_signal = pd.read_csv(BASE_PATH + 'load.csv')
# %% load map of europe
map_img = mpimg.imread('europe_map.png')
# %%
generator.columns

edges.columns

nodes.columns

load_signal.columns

load_signal['1']
# %% generate graph (there are 1494 nodes in total)


graph = nx.Graph()

# to import nodes to graph one needs a list of IDs and to import edges one need a list of tuples
node_list = nodes['ID'].tolist()
edge_list_start_ID = edges['fromNode'].tolist()
edge_list_end_ID = edges['toNode'].tolist()
edge_list = list(zip(edge_list_start_ID, edge_list_end_ID))
edge_weights = 1 / edges['length']  # larger length = smaller weight
weighted_edge_list = list(zip(edge_list_start_ID, edge_list_end_ID, edge_weights))

# lon = x, lat = y - for plotting graph on map
lon = nodes['longitude']
lat = nodes['latitude']

edge_list_lon = np.zeros((2, len(edge_list_start_ID)))
edge_list_lat = np.zeros((2, len(edge_list_start_ID)))
for i in range(len(edge_list_start_ID)):
    edge_list_lon[0, i] = nodes[nodes['ID'] == edges.iloc[i]['fromNode']]['longitude'].to_numpy()
    edge_list_lon[1, i] = nodes[nodes['ID'] == edges.iloc[i]['toNode']]['longitude'].to_numpy()
    edge_list_lat[0, i] = nodes[nodes['ID'] == edges.iloc[i]['fromNode']]['latitude'].to_numpy()
    edge_list_lat[1, i] = nodes[nodes['ID'] == edges.iloc[i]['toNode']]['latitude'].to_numpy()

graph.add_nodes_from(node_list)
# graph.add_edges_from(edge_list)  # constructing unweighted graph
graph.add_weighted_edges_from(weighted_edge_list)  # constructing weighted graph according to length of edges (lines)

# %% plot graph
plt.figure()
plot_map(map_img)
plot_signal_on_graph(lon, lat, title='Graph representation')
plot_power_lines(edge_list_lon, edge_list_lat)
plt.show()

# 3. Exploration --------------------------------------------------------------
# %% connected components
conn_comp = nx.number_connected_components(graph)
print(f'Number of connected components: {conn_comp}')
# %% sparsity
adjacency = nx.to_numpy_matrix(graph)  # 'to_numpy_matrix' returns the adjacency matrix of G as a numpy matrix
adjacency /= np.max(adjacency)  # if weighted graph to normalise
plt.figure
plt.spy(adjacency, markersize=0.5)
plt.show()
# %% diameter
diam = nx.diameter(graph)
print(f'Diameter: {diam}')
#
# would be interesting to know what is the largest shortest path is kilometers
#
#

# %% degree distribution (looks like power law)
degrees = np.count_nonzero(adjacency, axis=1)
plt.figure
plt.hist(degrees)
plt.title('Degree distribution')
plt.show()
# %% spectrum
# computing normalized Laplacian
laplacian = compute_laplacian(adjacency, normalize=True)  # adjacency matrix needs to be unweighted!
# spectral decomposition
lam, U = spectral_decomposition(laplacian)

plot_fig = False  # set false to not plot eigenfunctions
if plot_fig is True:
    plt.figure(figsize=(18, 9))
    plt.subplot(231)
    plot_signal_on_graph(lon, lat, x=U[:, 0], title='Eigenfunction #0 $\lambda$ =' + str(float('%.1g' % lam[0])))
    plot_power_lines(edge_list_lon, edge_list_lat)
    plt.subplot(232)
    plot_signal_on_graph(lon, lat, x=U[:, 1], title='Eigenfunction #1 $\lambda$ =' + str(float('%.1g' % lam[1])))
    plot_power_lines(edge_list_lon, edge_list_lat)
    plt.subplot(233)
    plot_signal_on_graph(lon, lat, x=U[:, 2], title='Eigenfunction #2 $\lambda$ =' + str(float('%.1g' % lam[2])))
    plot_power_lines(edge_list_lon, edge_list_lat)
    plt.subplot(234)
    plot_signal_on_graph(lon, lat, x=U[:, 3], title='Eigenfunction #3 $\lambda$ =' + str(float('%.1g' % lam[3])))
    plot_power_lines(edge_list_lon, edge_list_lat)
    plt.subplot(235)
    plot_signal_on_graph(lon, lat, x=U[:, 10], title='Eigenfunction #10 $\lambda$ =' + str(float('%.1g' % lam[10])))
    plot_power_lines(edge_list_lon, edge_list_lat)
    plt.subplot(236)
    plot_signal_on_graph(lon, lat, x=U[:, 100], title='Eigenfunction #100 $\lambda$ =' + str(float('%.1g' % lam[100])))
    plot_power_lines(edge_list_lon, edge_list_lat)
    plt.show()

# %% define and plot load signal
load = load_signal.to_numpy()[:, 1:]
# average the load signals
mean_load = np.mean(load, axis=0)
mean_load = mean_load.reshape(len(mean_load), -1)

# plot averaged load signal on graph (adjust vlim if necessary, vlim are the clip values of the signal for representation)
plt.figure()
plot_map(map_img)
plot_signal_on_graph(lon, lat, x=mean_load, title='Average load [MWh]')
plot_power_lines(edge_list_lon, edge_list_lat)
plt.show()

# %% plotting spectrum of the average load signal
plt.figure
plt.plot(lam, np.abs(GFT(U, mean_load)))
plt.xlabel('$\lambda$')
plt.ylabel('GFT')
plt.title('Spectrum of the average load signal')
plt.show()
# %% clustering coefficient
clustering_coeff = nx.average_clustering(graph)
print(f'Clustering coefficient: {clustering_coeff}')

# %% fourier transform of time series
load_node = load_signal.to_numpy()[:, 4].astype(float)
n_sample = len(load_node)
x = np.arange(n_sample)
sampling_frequency = 1  # because we sample once an hour -> 24 times a day
load_node_hat = np.fft.fft(load_node)
f = np.arange(0, len(load_node_hat) // 2, 1) * sampling_frequency / n_sample  # frequencies for the x axis
plt.figure(1)
ax = plt.gca()
ax.plot(f, np.abs(load_node_hat)[:len(load_node) // 2], c='red')
ax.scatter([1 / (365 * 24), 1 / (7 * 24), 1 / 24, 1 / 12], [100000, 100000, 100000, 100000], c='green',
           marker='o')  # plotting green points at frequencies of 1 day, 1 week and 1 year arbitrary height
plt.show()
print(np.sum(np.abs(load_node_hat)))

# %%
# creat signals on graph called day, week, year which are the frequency magnitudes of the Fourier decomposition
import importlib
import functions

importlib.reload(functions)

numpy_load_signal = load_signal.to_numpy()[:, 1:].astype(float)
half_day = np.zeros((numpy_load_signal.shape[1], 1))
day = np.zeros((numpy_load_signal.shape[1], 1))
week = np.zeros((numpy_load_signal.shape[1], 1))
year = np.zeros((numpy_load_signal.shape[1], 1))
for i in range(numpy_load_signal.shape[1]):
    load_node = numpy_load_signal[:, i]
    temp_half_day, temp_day, temp_week, temp_year = functions.magnitude_getter(load_node)
    half_day[i] = temp_half_day
    day[i] = temp_day
    week[i] = temp_week
    year[i] = temp_year

half_day[(np.isnan(half_day))] = 0
day[(np.isnan(day))] = 0
week[(np.isnan(week))] = 0
year[(np.isnan(year))] = 0
# %%
# plotting day, week and year signals on graph

plt.figure(figsize=(18, 9))
plt.subplot(221)
plot_map(map_img)
plot_signal_on_graph(lon, lat, x=half_day, title='Amplitude of half day frequency')
plot_power_lines(edge_list_lon, edge_list_lat)

plt.subplot(222)
plot_map(map_img)
plot_signal_on_graph(lon, lat, x=day, title='Amplitude of once a day frequency')
plot_power_lines(edge_list_lon, edge_list_lat)

plt.subplot(223)
plot_map(map_img)
plot_signal_on_graph(lon, lat, x=week, title='Amplitude of once a week frequency')
plot_power_lines(edge_list_lon, edge_list_lat)

plt.subplot(224)
plot_map(map_img)
plot_signal_on_graph(lon, lat, x=year, title='Amplitude of once a year frequency')
plot_power_lines(edge_list_lon, edge_list_lat)
plt.show()

# %%
# generator graph (the IDs of the generators do not correspond to the IDs of the nodes.
geny = generator.to_numpy()
nod = nodes.to_numpy()
print(np.max(geny[:, 0]))
print(geny[np.argmin(geny[:, 0]), 0:3])
nod[0:130, 0:3]

# %%
graph_generator = nx.Graph()

# to import nodes to graph one needs a list of IDs and to import edges one need a list of tuples
node_list = generator['ID'].tolist()

# lon = x, lat = y - for plotting graph on map
lon_g = generator['longitude']
lat_g = generator['latitude']

graph.add_nodes_from(node_list)

# %%
geny = generator.to_numpy()

plt.figure(1)


def plot_generator(geny):
    types = np.unique(geny[:, 7])
    colours = matplotlib.colors.ListedColormap(['k', 'b', 'r', 'y', 'g', 'magenta', 'chocolate', 'cyan', 'indigo'])
    generator_type = dict(zip(types, np.arange(10)))
    sizes = (geny[:, 9] / 20)
    node_list = geny[:, 0]
    fig = plt.gcf()
    ax = plt.gca()
    scatter = ax.scatter(geny[:, 5], geny[:, 4], c=list(map(generator_type.get, geny[:, 7])), cmap=colours,
                         s=list(sizes))
    legend1 = plt.legend(handles=scatter.legend_elements()[0], labels=list(types), title="Fuel type")
    ax.add_artist(legend1)
    plt.show()


plt.figure()
plot_power_lines(edge_list_lon, edge_list_lat)
plot_map(map_img)
plot_generator(geny)

# %% load forecast data (only run this if really necessary - it takes for ever)
run_this_bit = False  # set true if you need to generate the csv file again
if run_this_bit is True:
    # get list of all 2192 folders containing the forcast data
    BASE_PATH_FORECAST = BASE_PATH + 'Nodal_FC/'
    folders = [x[0] for x in os.walk(BASE_PATH_FORECAST)][1:]  # [1:] removes the parent directory
    # extract the forecast data from all folders
    for i, base in enumerate(folders):
        # get the first 12h of every forecast
        tmp_solar_fc = pd.read_csv(base + '/solar_forecast.csv').to_numpy()[:12, 1:]
        tmp_wind_fc = pd.read_csv(base + '/wind_forecast.csv').to_numpy()[:12, 1:]
        print(f'running... {np.round(i / len(folders) * 100, decimals=2)}%')
        # stack them on top of each other
        if i == 0:
            solar_fc = tmp_solar_fc
            wind_fc = tmp_wind_fc
        else:
            solar_fc = np.append(solar_fc, tmp_solar_fc, axis=0)
            wind_fc = np.append(wind_fc, tmp_wind_fc, axis=0)
    print('done!')
    # save generated data as csv file for fast reloading
    np.savetxt(BASE_PATH + 'solar_fc.csv', solar_fc, delimiter=',')
    np.savetxt(BASE_PATH + 'wind_fc.csv', wind_fc, delimiter=',')
# %% load forecast data from single csv file (run this instead of the above if the csv file is already generated!)
solar_fc = np.loadtxt(BASE_PATH + 'solar_fc.csv', delimiter=',')
wind_fc = np.loadtxt(BASE_PATH + 'wind_fc.csv', delimiter=',')
# %% load power capacities
# directly convert to numpy and extract the proportional capacities
solar_cp = pd.read_csv(BASE_PATH + 'solar_layouts_COSMO.csv').to_numpy()[:, 1]
wind_cp = pd.read_csv(BASE_PATH + 'wind_layouts_COSMO.csv').to_numpy()[:, 1]
# %% load actual data
solar_ts_complete = pd.read_csv(BASE_PATH + 'solar_signal_COSMO.csv').to_numpy()
solar_ts = solar_ts_complete[:, 1:].astype(float)
wind_ts = pd.read_csv(BASE_PATH + 'wind_signal_COSMO.csv').to_numpy()[:, 1:].astype(
    float)  # directly convert into numpy and remove the time column
# get time vector of the signals (same for all) - format: 'YYYY-MM-DD HH:MM:SS'
time_vector = solar_ts_complete[:, 0]
# %% convert signals to MWh
solar_fc_MWh = solar_fc * solar_cp
solar_ts_MWh = solar_ts * solar_cp
wind_fc_MWh = wind_fc * wind_cp
wind_ts_MWh = wind_ts * wind_cp
# %% calculate relative percentage difference (RPD) between forecast and actual
solar_diff = RPD(solar_fc_MWh, solar_ts_MWh)
wind_diff = RPD(wind_fc_MWh, wind_ts_MWh)
# %% plot data for first week of 'node'
node = 100
start_time = time_vector[0]
end_time = time_vector[7 * 24]
plot_forecast_actual(solar_fc_MWh, solar_ts_MWh, wind_fc_MWh, wind_ts_MWh, time_vector, start_time, end_time, node)
# %% plot the average solar and wind energy and the average forecasting error
plot_forecasting_on_graph(solar_fc_MWh, solar_ts_MWh, solar_diff, wind_fc_MWh, wind_ts_MWh, wind_diff, lon, lat,
                          edge_list_lon, edge_list_lat, map_img)
# %% machine learning
# initialize model

#net = NeuralNet(in_size=12, out_size=12)  # simple net
#net = ConvNet()  # conv net
net = GCN(adjacency, np.diag(np.ravel(np.sum(adjacency, axis=1))))

# get name of the net used
model_name = net.__class__.__name__

# %% prepare training and test data
if model_name != 'GCN':
    solar_fc_tensor = get_nn_inputs(solar_fc_MWh)
    solar_ts_tensor = get_nn_inputs(solar_ts_MWh)
    wind_fc_tensor = get_nn_inputs(wind_fc_MWh)
    wind_ts_tensor = get_nn_inputs(wind_ts_MWh)
else:
    solar_fc_tensor = torch.Tensor(solar_fc_MWh)
    solar_ts_tensor = torch.Tensor(solar_fc_MWh)
    wind_fc_tensor = torch.Tensor(wind_fc_MWh)
    wind_ts_tensor = torch.Tensor(wind_fc_MWh)

solar_train_feat, solar_train_target, solar_test_feat, solar_test_target = train_test_set(solar_fc_tensor,
                                                                                          solar_ts_tensor)
wind_train_feat, wind_train_target, wind_test_feat, wind_test_target = train_test_set(wind_fc_tensor, wind_ts_tensor)

solar_train_feat_std, mean_solar, std_solar = standardize(solar_train_feat)
solar_test_feat_std = fit_standardize(solar_test_feat, mean_solar, std_solar)
wind_train_feat_std, mean_wind, std_wind = standardize(wind_train_feat)
wind_test_feat_std = fit_standardize(wind_test_feat, mean_wind, std_wind)

# %% for conv model
if model_name == 'ConvNet':
    solar_train_feat_std = solar_train_feat_std.unsqueeze(1)
    solar_test_feat_std = solar_test_feat_std.unsqueeze(1)
    
    wind_train_feat_std = wind_train_feat_std.unsqueeze(1)
    wind_test_feat_std = wind_test_feat_std.unsqueeze(1)

# %% train the model for solar engergy (only do with small amount of data)
train_loss, test_loss = train(model=net, train_inputs=solar_train_feat_std, train_targets=solar_train_target, 
                              test_inputs=solar_test_feat_std, test_targets=solar_test_target, n_epoch=50, batch_size=15)
plt.plot(train_loss, label='train_loss')
plt.plot(test_loss, label='test_loss')
# %% test the trained model
node = 1000
start_h = 5000
duration_h = 5*24


model_wind = torch.load('NeuralNet_trained_GCN_wind.pt')

model_solar = torch.load('NeuralNet_trained_GCN_solar.pt')
    
    
if model.__class__.__name__ != 'GCN':
    pred = retrieve_ts_from_nn_outputs(model(solar_test_feat_std))
    forecast = retrieve_ts_from_nn_outputs(solar_test_feat)
    target = retrieve_ts_from_nn_outputs(solar_test_target)
    plt.plot(pred[7:14*24,1300], label='pred')
    plt.plot(target[7:14*24,1300], label='target')
    plt.plot(forecast[7:14*24,1300], label='forecast')
    plt.legend()
    plt.show()
else:
    plt.subplot(221)
    x = wind_test_feat_std.clone()
    forecast = wind_test_target.detach().numpy()

    pred = model_wind(x).detach().numpy()
    plt.plot(pred[start_h:start_h+duration_h,node], label='pred')
    plt.plot(forecast[start_h:start_h+duration_h,node], label='forecast')
    plt.legend()
    plt.title(f'Wind Forecast Prediction of Node {node} from {time_vector[start_h]} to {time_vector[start_h+duration_h]}')
    plt.xlabel('Time [h]')
    plt.ylabel('MWh')
    plt.show()
    
    plt.subplot(223)
    rpd_wind = np.abs(RPD(pred, forecast))
    plt.plot(rpd_wind[start_h:start_h+duration_h,node], label='RPD')
    plt.plot(np.full(rpd_wind[start_h:start_h+duration_h,node].shape, np.mean(rpd_wind[start_h:start_h+duration_h,node])), label='Mean')
    plt.legend()
    plt.title('Relative Percentage Difference of Wind Forecast Prediction')
    plt.xlabel('Time [h]')
    plt.ylabel('%')
    
    plt.subplot(222)
    x = solar_test_feat_std.clone()
    forecast = solar_test_target.detach().numpy()

    pred = model_solar(x).detach().numpy()
    plt.plot(pred[start_h:start_h+duration_h,node], label='pred')
    plt.plot(forecast[start_h:start_h+duration_h,node], label='forecast')
    plt.legend()
    plt.title(f'Solar Forecast Prediction of Node {node} from {time_vector[start_h]} to {time_vector[start_h+duration_h]}')
    plt.xlabel('Time [h]')
    plt.ylabel('MWh')
    plt.show()
    
    plt.subplot(224)
    rpd_solar = np.abs(RPD(pred, forecast))
    rpd_solar[rpd_solar == 200] = 0 # removing night error (not interesting)
    plt.plot(rpd_solar[start_h:start_h+duration_h,node], label='RPD')
    plt.plot(np.full(rpd_solar[start_h:start_h+duration_h,node].shape, np.mean(rpd_solar[start_h:start_h+duration_h,node])), label='Mean')
    plt.legend()
    plt.title('Relative Percentage Difference of Solar Forecast Prediction')
    plt.xlabel('Time [h]')
    plt.ylabel('%')
    plt.show()

# %% analyze forecasting prediction on the graph
rpd_solar_avg = np.mean(rpd_solar, axis=0)
rpd_wind_avg = np.mean(rpd_wind, axis=0)

plt.figure()
plot_map(map_img)
plot_signal_on_graph(lon, lat, rpd_solar_avg, title='Average Solar Forecasting Prediction Error [%]')
plot_power_lines(edge_list_lon, edge_list_lat)
plt.show()

plt.figure()
plot_map(map_img)
plot_signal_on_graph(lon, lat, rpd_wind_avg, title='Average Wind Forecasting Prediction Error [%]')
plot_power_lines(edge_list_lon, edge_list_lat)
plt.show()

# %% analize forecasting prediction vs node degree
plt.subplot(121)
plt.title('Average Solar Forecasting Prediction Error vs Node Degree')
x = np.ravel(degrees)
p = np.polyfit(x, rpd_solar_avg, 1)
plt.scatter(x, rpd_solar_avg)
y = np.append(degrees, np.ones(degrees.shape), axis=1).dot(p)
plt.plot(x, np.ravel(y), 'r', label='Linear fit')
plt.xlabel('degree')
plt.ylabel('Error [%]')
plt.legend()
plt.subplot(122)
plt.title('Average Wind Forecasting Prediction Error vs Node Degree')
p = np.polyfit(x, rpd_wind_avg, 1)
plt.scatter(x, rpd_wind_avg)
y = np.append(degrees, np.ones(degrees.shape), axis=1).dot(p)
plt.plot(x, np.ravel(y), 'r', label='Linear fit')
plt.xlabel('degree')
plt.ylabel('Error [%]')
plt.legend()
plt.show()

# %% distance between two closest and furthest nodes
latitudes = nodes['latitude'].to_numpy()
longitudes = nodes['longitude'].to_numpy()
positions = np.concatenate((longitudes[:, np.newaxis], latitudes[:, np.newaxis]), axis=1)
earth_radius = 6367449  # in meters
shortest_dist = 1e9
largest_dist = 0
# %% takes a long time
for num,i in enumerate(positions):
    if num % 10 == 0:
        print(num)
    for j in positions[num+1:,:]:
        if not (i == j).all():
            dist= geopy.distance.distance(i, j).km
            if dist < shortest_dist:
                shortest_dist = dist
            if dist > largest_dist:
                largest_dist = dist
# shortest_dist = 0.26449321005593207
# largest_dist = 4283.281141199654

# %%



# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import functions
from functions import *
import networkx as nx

# %%
BASE_PATH = 'P:/Python/Network_tour/Data_for_project/'  # adjust if necessary

edges = pd.read_csv(BASE_PATH + 'edges.csv')
nodes = pd.read_csv(BASE_PATH + 'nodes.csv')
generator = pd.read_csv(BASE_PATH + 'generator.csv')
load_signal = pd.read_csv(BASE_PATH + 'load.csv')
# %%
generator.columns

edges.columns

nodes.columns

load_signal.columns

load_signal['1']
# %% generate graph


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

plot_fig = True  # set false to not plot eigenfunctions
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
# extract load signal from start_hour to end_hour
start_hour = 0
end_hour = 23
load_day1 = load_signal.iloc[start_hour:end_hour + 1].to_numpy()[:, 1:]
# average the load signals
mean_load_day1 = np.mean(load_day1, axis=0)
mean_load_day1 /= np.max(mean_load_day1)
mean_load_day1 = mean_load_day1.reshape(len(mean_load_day1), -1)

# plot averaged load signal on graph (adjust vlim if necessary, vlim are the clip values of the signal for representation)
plt.figure()
plot_signal_on_graph(lon, lat, x=mean_load_day1, title='Normalised load over one day', vlim=(0, 0.06))
plot_power_lines(edge_list_lon, edge_list_lat)
plt.show()

# %% plotting spectrum of the average load signal
plt.figure
plt.plot(lam, np.abs(GFT(U, mean_load_day1)))
plt.xlabel('$\lambda$')
plt.ylabel('GFT')
plt.title('Spectrum of the average load signal')
plt.show()
# %% clustering coefficient
clustering_coeff = nx.average_clustering(graph)
print(f'Clustering coefficient: {clustering_coeff}')

# %% fourier transform of time series
load_node = load_signal.to_numpy()[:, 4]
n_sample=len(load_node)
x = np.arange(n_sample)
sampling_frequency = 1 # because we sample once an hour -> 24 times a day
load_node_hat = np.fft.fft(load_node)
f = np.arange(0, len(load_node_hat) // 2, 1) * sampling_frequency / n_sample # frequencies for the x axis
plt.figure(1)
ax = plt.gca()
ax.plot(f, np.abs(load_node_hat)[:len(load_node) // 2], c='red')
ax.scatter([1/(365*24), 1/(7*24), 1/24,1/12], [100000, 100000, 100000,100000], c='green', marker='o') # plotting green points at frequencies of 1 day, 1 week and 1 year arbitrary height
plt.show()
print(np.sum(np.abs(load_node_hat)))

# %%
# creat signals on graph called day, week, year which are the frequency magnitudes of the Fourier decomposition
import importlib
import functions
importlib.reload(functions)

numpy_load_signal=load_signal.to_numpy()[:,1:]
half_day=np.zeros((numpy_load_signal.shape[1],1))
day=np.zeros((numpy_load_signal.shape[1],1))
week=np.zeros((numpy_load_signal.shape[1],1))
year=np.zeros((numpy_load_signal.shape[1],1))
for i in range(numpy_load_signal.shape[1]):
    load_node = numpy_load_signal[:, i]
    temp_half_day, temp_day, temp_week, temp_year=functions.magnitude_getter(load_node)
    half_day[i]=temp_half_day
    day[i]=temp_day
    week[i]=temp_week
    year[i]=temp_year

half_day[(np.isnan(half_day))]=0
day[(np.isnan(day))]=0
week[(np.isnan(week))]=0
year[(np.isnan(year))]=0
# %%
# plotting day, week and year signals on graph

plt.figure(figsize=(18, 9))
plt.subplot(221)
plot_signal_on_graph(lon, lat, x=half_day, title='Fourier component of load time series corresponding to once every half day frequency', vlim=(0, np.max(half_day)))
plot_power_lines(edge_list_lon, edge_list_lat)

plt.subplot(222)
plot_signal_on_graph(lon, lat, x=day, title='Fourier component of load time series corresponding to once a day frequency', vlim=(0, np.max(day)))
plot_power_lines(edge_list_lon, edge_list_lat)

plt.subplot(223)
plot_signal_on_graph(lon, lat, x=week, title='Fourier component of load time series corresponding to once a week frequency', vlim=(0, np.max(week)))
plot_power_lines(edge_list_lon, edge_list_lat)

plt.subplot(224)
plot_signal_on_graph(lon, lat, x=year, title='Fourier component of load time series corresponding to once a year frequency', vlim=(0, np.max(year)))
plot_power_lines(edge_list_lon, edge_list_lat)
plt.show()


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import *

# %%
BASE_PATH = 'data/'  # adjust if necessary

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
import networkx as nx
G=nx.Graph()

# to import nodes to G one needs a list of IDs and to import edges one need a list of tupples
node_list=nodes['ID'].tolist()
edge_list_start_ID=edges['fromNode'].tolist()
edge_list_end_ID=edges['toNode'].tolist()
edge_list=list(zip(edge_list_start_ID,edge_list_end_ID))
edge_weights=1/edges['length']  # larger lenght = smaller weight
weighted_edge_list=list(zip(edge_list_start_ID,edge_list_end_ID,edge_weights))

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
    

G.add_nodes_from(node_list)
G.add_edges_from(edge_list)  # constructing unweighted graph
#G.add_weighted_edges_from(weighted_edge_list)  # constructing weighted graph according to length of edges (lines)

# %% plot graph
plt.figure()
plot_signal_on_graph(lon, lat, title='Graph representation')
plot_power_lines(edge_list_lon, edge_list_lat)
plt.show()

# 3. Exploration --------------------------------------------------------------
# %% connected components
conn_comp = nx.number_connected_components(G)
print(f'Number of connected components: {conn_comp}')
# %% sparsity
adjacency = nx.to_numpy_matrix(G)  # 'to_numpy_matrix' returns the adjacency matrix of G as a numpy matrix
#adjacency /= np.max(adjacency)  # if weighted graph
plt.figure
plt.spy(adjacency, markersize=0.5)
plt.show()
# %% diameter
diam = nx.diameter(G)
print(f'Diameter: {diam}')
# %% degree distribution (looks like power law)
degrees = np.count_nonzero(adjacency, axis=1)
plt.figure
plt.hist(degrees)
plt.title('Degree distribution')
plt.show()
# %% spectrum
# computing normalized Laplacian
laplacian = compute_laplacian(adjacency, normalize=True)  # adjacency matrix need to be unweighted!
# spectral decomposition
lam, U = spectral_decomposition(laplacian)

plotfig = True  # set false to not plot eigenvector visualisation
if plotfig is True:
    plt.figure(figsize=(18, 9))
    plt.subplot(231)
    plot_signal_on_graph(lon, lat, x=U[:,0], title='Eigenvector #0')
    plot_power_lines(edge_list_lon, edge_list_lat)
    plt.subplot(232)
    plot_signal_on_graph(lon, lat, x=U[:,1], title='Eigenvector #1')
    plot_power_lines(edge_list_lon, edge_list_lat)
    plt.subplot(233)
    plot_signal_on_graph(lon, lat, x=U[:,2], title='Eigenvector #2')
    plot_power_lines(edge_list_lon, edge_list_lat)
    plt.subplot(234)
    plot_signal_on_graph(lon, lat, x=U[:,3], title='Eigenvector #3')
    plot_power_lines(edge_list_lon, edge_list_lat)
    plt.subplot(235)
    plot_signal_on_graph(lon, lat, x=U[:,10], title='Eigenvector #10')
    plot_power_lines(edge_list_lon, edge_list_lat)
    plt.subplot(236)
    plot_signal_on_graph(lon, lat, x=U[:,100], title='Eigenvector #100')
    plot_power_lines(edge_list_lon, edge_list_lat)

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
clustering_coeff = nx.average_clustering(G)
print(f'Clustering coefficient: {clustering_coeff}')




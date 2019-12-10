# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
edges = pd.read_csv('edges.csv')
nodes = pd.read_csv('nodes.csv')
generator = pd.read_csv('generator.csv')
load_signal = pd.read_csv('load.csv')
# %%
generator.columns

edges.columns

nodes.columns

load_signal.columns

load_signal['1']
# %%
import networkx as nx
G=nx.Graph()

# to import nodes to G one needs a list of IDs and to import edges one need a list of tupples
node_list=nodes['ID'].tolist()
edge_list_start=edges['fromNode'].tolist()
edge_list_end=edges['toNode'].tolist()
edge_list=list(zip(edge_list_start,edge_list_end))

G.add_nodes_from(node_list)
G.add_edges_from(edge_list)

nx.draw(G)
plt.show()

#number of connected components
nx.number_connected_components(G)
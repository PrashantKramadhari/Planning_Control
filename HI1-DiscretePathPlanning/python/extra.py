#!/usr/bin/env python
# coding: utf-8

# # TSFS12 Hand-in exercise 1, extra assignment: Discrete planning in structured road networks
# Erik Frisk (erik.frisk@liu.se)

# Do initial imports of packages needed
In[1]:


import matplotlib.pyplot as plt
from misc import Timer, LatLongDistance
from osm import loadOSMmap


# Activate plots in external windows (needed for mission definition)




# # Load map, define state transition function, and heuristic for Astar


dataDir = '../Maps/'
osmFile = 'linkoping.osm'
figFile = 'linkoping.png'
osmMap = loadOSMmap(dataDir + osmFile, dataDir + figFile)

num_nodes = len(osmMap.nodes)  # Number of nodes in the map


# Define function to compute possible next nodes from node x and the corresponding distances distances.


def f_next(x):
    """Compute, neighbours for a node"""
    cx = osmMap.distancematrix[x, :].tocoo()
    return cx.col, np.full(cx.col.shape, np.nan), cx.data



def heuristic(x, xg):
    # YOUR CODE HERE
    p_x = osmMap.nodeposition[x]
    p_g = osmMap.nodeposition[xg]
    return 0.0


# # Define planning missions with same goal node

# Predefined planning missions with the same goal node that you can use. You are welcome to experiment with other missions.


missions = [
    {'start': {'id': 10906}, 'goal': {'id': 1024}},
    {'start': {'id': 3987}, 'goal': {'id': 1024}},
    {'start': {'id': 423}, 'goal': {'id': 1024}}]


# # Exercises






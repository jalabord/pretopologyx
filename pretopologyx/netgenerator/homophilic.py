import numpy as np
import math, random


# data is a list of agents where every column is an attribute
# label has all the names of the attributes
# attribute
# hl is the homophily level: if uniform() > hl = random connection, else connect to closest not yet connected
# influence should also be a parameter
# we need to better think about edges weight, for the moment only one is possible (is influence for this case)
def homophilic_network(pre_space, homophily_attribute, hl, avg_connectivity, influence):

    adj = np.zeros((pre_space.size, pre_space.size))
    ht = np.argwhere(np.array(pre_space.attribute_labels) == homophily_attribute)[0, 0]

    # Each row is an agent with the columns stocking the index of their most homopĥilic neighbors
    matrix_homophilic_neighbors = np.zeros((pre_space.size, pre_space.size))
    for i in range(pre_space.size):
        hom_agent = np.full(pre_space.size, pre_space.environment[i, ht])
        hom_diffs = np.absolute(hom_agent - pre_space.environment[:, ht])
        matrix_homophilic_neighbors[i, :] = np.copy(hom_diffs.argsort())

    for i in range(0, math.floor(pre_space.size*avg_connectivity/2)):
        rand_index = random.randint(0, pre_space.size-1)
        index_neigh = 0
        rand = random.uniform(0, 1)
        if rand < hl:
            index_neigh = matrix_homophilic_neighbors[rand_index, len(list(G.neighbors(rand_index)))+1]
        else:
            index_neigh = random.randint(0, pre_space.size-1)
            # We check for an undirected graph if they are already neighbors
            while adj[rand_index, index_neigh] or rand_index == index_neigh:
                index_neigh = random.randint(0, pre_space.size-1)
        adj[rand_index, index_neigh] = random.uniform(0, influence)
    return adj

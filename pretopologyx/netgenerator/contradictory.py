import numpy as np
import math, random
import networkx as nx


# generates a network using the generation method and its arguments,
# and takes care not to connect to neighbors in the contradictory networks
# this is useful if we are interested in creating a networks of friends and enemies,
# or networks for different geographical locations, for example.
def contradictory_network(generation_metod, contradictory_networks, *args):

    G = nx.Graph()

    for agent_id in range(args[0]):
        G.add_node(agent_id)

    # for i in range(0, math.floor(pop_size * average_enemy_connectivity / 2)):
    #     rand_index = random.randint(0, pop_size - 1)
    #     index_neigh = random.randint(0, pop_size - 1)

    # We check for an undirected graph if they are already neighbors or if they are friends
    # while G.has_edge(rand_index, index_neigh) or G_enemy.has_edge(rand_index,
    #                                                              index_neigh) or rand_index == index_neigh:
    #    index_neigh = random.randint(0, pop_size - 1)
    #    G_enemy.add_edge(rand_index, index_neigh, influence=random.uniform(0, 0.23))

    # print("We finished the enemy network")
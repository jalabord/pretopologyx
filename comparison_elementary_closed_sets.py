import numpy as np
import networkx as nx
import time
import sys

from pretopologyx.space.pretopological_space import PretopologicalSpace, Prenetwork
from pretopologyx.structure.closures import closures, minimal_closed_sets, elementary_closures
from pretopologyx.structure.hierarchy import elementary_closed_hierarchy
import pretopologyx.lsp.biggest_lower_dnf as bldnf


def comparison():

    prenets_strow = strowgatz_prespace(2000)
    prenets_barab = barabasi_albert_prespace(2000)
    pre_space = PretopologicalSpace(prenets_strow + prenets_barab, [[0, 1], [2, 3], [4, 5]])
    initial_set = np.ones(pre_space.size)

    start = time.time()
    list_closures_old = closures(pre_space, [initial_set])
    stop = time.time()
    print("the old method took: " + str(stop - start))

    start = time.time()
    list_closures_new = elementary_closures(pre_space, degree=1)
    stop = time.time()
    print("the new method took: " + str(stop - start))

    print("the number of different closures was: " + str(len(list_closures_new)))

    # We test if both methods return the same results
    # It's printing True chen one is empty and the other no...
    are_equal = compare_closures(list_closures_old, list_closures_new)
    print(are_equal)

    sys.exit()


# We compare both closures just to be sure
def compare_closures(old_closures, new_closures):
    equal = True
    old_array = np.array(old_closures)
    for cl in new_closures:
        if not (cl == old_array).all(axis=1).any():
            equal = False
    return equal


def barabasi_albert_prespace(size):
    G1 = nx.barabasi_albert_graph(size, 10)
    adj1 = add_random_weights(np.array(nx.adjacency_matrix(G1).todense()))
    prenet1 = Prenetwork(adj1, [0.2])

    G2 = nx.barabasi_albert_graph(size, 10)
    adj2 = add_random_weights(np.array(nx.adjacency_matrix(G2).todense()))
    prenet2 = Prenetwork(adj2, [0.2])

    return [prenet1, prenet2]


def geometric_prespace(size):
    G1 = nx.watts_strogatz_graph(size, 6, 0.1)
    adj1 = np.array(nx.adjacency_matrix(G1).todense())
    prenet1 = Prenetwork(adj1, [0.7])
    pre_space = PretopologicalSpace([prenet1], [[0]])
    return pre_space


# When having 2000 and [[0], [1], [2], [3]], we pass from 213s to 186s
# When having 2000 and [[0,1], [2,3]], we pass from 986s to 273s !!, diff closures: 872
def strowgatz_prespace(size):
    G1 = nx.watts_strogatz_graph(size, 6, 0.1)
    adj1 = add_random_weights(np.array(nx.adjacency_matrix(G1).todense()))
    prenet1 = Prenetwork(adj1, [0.7])
    G2 = nx.watts_strogatz_graph(size, 6, 0.1)
    adj2 = add_random_weights(np.array(nx.adjacency_matrix(G2).todense()))
    prenet2 = Prenetwork(adj2, [0.7])
    G3 = nx.watts_strogatz_graph(size, 6, 0.1)
    adj3 = add_random_weights(np.array(nx.adjacency_matrix(G3).todense()))
    prenet3 = Prenetwork(adj3, [0.7])
    G4 = nx.watts_strogatz_graph(size, 6, 0.1)
    adj4 = add_random_weights(np.array(nx.adjacency_matrix(G4).todense()))
    prenet4 = Prenetwork(adj4, [0.7])

    return [prenet1, prenet2, prenet3, prenet4]


def random_prespace(size):
    random_net = np.random.choice(2, size**2).reshape(size, size)
    prenet = Prenetwork(random_net, [0.7])
    pre_space = PretopologicalSpace([prenet], [[0]])
    return pre_space


def add_random_weights(adj):
    weights = np.random.rand(len(adj), len(adj))
    return adj*weights


if __name__ == '__main__':
    comparison()

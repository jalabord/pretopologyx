import numpy as np
import time
import itertools
from pretopologyx.space.metrics import pseudoclosure


# methods for elementary pseudoclosure of different degrees and minimal closed sets

def minimal_closed_sets(elementary_cls):
    minimal_cls = list()
    while elementary_cls:
        F = elementary_cls.pop()
        minimal = True
        temporary_cls = elementary_cls.copy()
        while temporary_cls and minimal:
            G = temporary_cls.pop()
            if ((F - G) >= 0).all() and not (F - G == 0).all():
                minimal = False
            elif ((F - G) <= 0).all():
                remove_array(elementary_cls, G)
        if minimal:
            minimal_cls.append(F)
    return minimal_cls


# args: the pretopological space and a list of the sets whose closure we will calculate
# the format of the initial set is an array of 0-1's
# returns a list of the closures of the initial sets in the same order
def closures(pre_space, initial_sets, elementary=True):
    list_closures = list()
    for init_set in initial_sets:
        if elementary:
            for idx, el in enumerate(init_set):
                if el == 1:
                    seed = np.zeros(pre_space.size)
                    seed[idx] = 1
                    list_closures.append(set_closure(pre_space, seed))
        else:
            list_closures.append(set_closure(pre_space, init_set))
    return list_closures


def set_closure(pre_space, initial_set):
    set_pseudoclosure = pseudoclosure(pre_space, initial_set)
    if int(sum(initial_set)) == int(sum(set_pseudoclosure)):
        return set_pseudoclosure
    else:
        return set_closure(pre_space, set_pseudoclosure)


def elementary_closures_shortest_degree(pre_space, degree=2):
    dict_closures = {i: set() for i in range(degree, pre_space.size + 1)}
    # we make a network with all the
    total_network = np.zeros((pre_space.size, pre_space.size))
    for prenetwork in pre_space.prenetworks:
        total_network += abs(prenetwork.network)
    for i in np.arange(pre_space.size):
        initial_set = np.zeros(pre_space.size, dtype='float')
        initial_set[i] = 1
        for dist in range(degree-1):
            # we need to be careful if we try create a set bigger than the component size
            row = total_network[i, :].copy()
            row[initial_set.astype('bool')] = 0
            ma = np.ma.masked_equal(row, 0.0, copy=False)
            i_new = np.argmin(ma)
            initial_set[i_new] = 1
        dict_closures[degree].add(initial_set.tostring())
    print(len(dict_closures[degree]))
    # Maybe we start from zero, in case sets smaller than degree are created
    for i in np.arange(degree, pre_space.size):
        counter = 0
        level_closed = set()
        while dict_closures[i]:
            counter += 1
            if counter % 1000 == 0:
                print(i)
            initial_set = np.fromstring(dict_closures[i].pop(), dtype='float')
            pseudoclosure_set = pseudoclosure(pre_space, initial_set)
            length = int(sum(pseudoclosure_set))
            cls_str = pseudoclosure_set.tostring()
            if length > i:
                if cls_str not in dict_closures[length]:
                    dict_closures[length].add(cls_str)
            else:
                level_closed.add(cls_str)
        dict_closures[i] = level_closed.copy()
    return create_closures_list(dict_closures, degree)


def elementary_closures_shortest_degree_largeron(pre_space, degree=2):
    # we make a network with all the
    total_network = np.zeros((pre_space.size, pre_space.size))
    for prenetwork in pre_space.prenetworks:
        total_network += abs(prenetwork.network)
    list_closures = list()
    for i in np.arange(pre_space.size):
        if i%200 == 0:
            print(i)
        initial_set = np.zeros(pre_space.size, dtype='float')
        initial_set[i] = 1
        for dist in range(degree-1):
            # we need to be careful if we try create a set bigger than the component size
            row = total_network[i, :].copy()
            row[initial_set.astype('bool')] = 0
            ma = np.ma.masked_equal(row, 0.0, copy=False)
            i_new = np.argmin(ma)
            initial_set[i_new] = 1
        list_closures.append(set_closure(pre_space, initial_set))
    return list_closures


# We transform from the dictionary that allowed us to improve Largeron's algorithm
def create_closures_list(dict_closures, degree):
    closures_list = list()
    noise_counter = 0
    for k, v in dict_closures.items():
        while v:
            temp = np.fromstring(v.pop(), dtype='float')

            # When working with clusters we eliminate the noise...
            if sum(temp) > degree + 3:
                closures_list.append(temp)
            else:
                noise_counter += 1

            # When comparing both closures algorithms we need all...
            # closures_list.append(temp)

    print("THE NOISE WAS: " + str(noise_counter))
    return closures_list


# We will try to change the list to a numpy array where we modify the values
def elementary_closures_to_test(pre_space, degree=2):
    print("this one")
    dict_closures = {i: np.zeros(pre_space.size) for i in range(degree, pre_space.size + 1)}
    list_temp = list()
    for i in np.arange(pre_space.size):
        for j, el in np.ndenumerate(pre_space.neighborhoods[0][i, :]):
            if el and j > i:
                initial_set = np.zeros(pre_space.size, dtype='float')
                initial_set[list([i, j[0]])] = 1
                list_temp.append(initial_set)
    for i in np.arange(degree, pre_space.size):
        print(i)
        counter = 0
        level_closed = list()
        while dict_closures[i]:
            counter += 1
            if counter % 1000 == 0:
                print(i)
            initial_set = dict_closures[i].pop()
            pseudoclosure_set = pseudoclosure(pre_space, initial_set)
            length = int(sum(pseudoclosure_set))
            if length > i and dict_closures[length]:
                print()
                if not (dict_closures[length] == pseudoclosure_set).any():
                    dict_closures[length] = np.vstack([dict_closures[length], pseudoclosure_set])
            else:
                level_closed.append(pseudoclosure_set)
        dict_closures[i] = np.vstack([dict_closures[i], level_closed.copy()])
    return dict_closures


# Improve this function
def remove_array(L, arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind], arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        print("wait")
        raise ValueError('array not found in list.')

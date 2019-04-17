import numpy as np
import itertools


def conjonctions_forbidden(pre_space, closures):
    conj_forb = np.array([])
    # each power of two represents a neighborhood
    neighs_used = np.power(2*np.ones(len(pre_space.network_index)), np.arange(len(pre_space.network_index)))
    for closure in closures:
        result = np.array([])
        susceptible = np.array(closure.flatten() == 0, dtype='float')
        infected = np.array(closure.flatten() == 1, dtype='float')
        # This is not considering weights for the moment
        for prenetwork in pre_space.prenetworks:
            neigh_infected = np.matmul(prenetwork.network, infected)
            susceptible_infected = susceptible*neigh_infected
            if result.size == 0:
                result = susceptible_infected
            else:
                result = np.column_stack([result, susceptible_infected])
        conj_temp = np.matmul(result, neighs_used)
        conj_forb = np.concatenate([conj_forb, conj_temp]) if conj_forb.size else conj_temp
        conj_forb = np.unique(conj_forb)
    # The conjunction forbidden is a number, it's decomposition in powers of two gives us the neighborhoods
    return conj_forb[conj_forb != 0]


# we definitely need to modify this
def estimate_dnf(conj_forb, number_neighs):
    fmt = '0' + str(number_neighs) + 'b'
    result = list()
    forbidden_conjs = list()
    for conj in conj_forb:
        # we identified the sets used from their number representation
        new_set = set()
        for el, i in enumerate(format(int(conj), fmt)[::-1]):
            if i == '1':
                new_set.add(el)
        subset = False
        for s in forbidden_conjs:
            if new_set.issubset(s):
                subset = True
                break
        if not subset:
            forbidden_conjs[:] = itertools.filterfalse(lambda x: x.issubset(new_set), forbidden_conjs)
            forbidden_conjs.append(new_set)
    # we now have the set of all the biggest forbidden conjonctions
    for j in range(1, number_neighs+1):
        for conj in itertools.combinations(np.arange(number_neighs), j):
            conj = set(conj)
            dont_add = False
            for forb_t in forbidden_conjs:
                if conj.issubset(forb_t):
                    dont_add = True
                    break
            if not dont_add:
                dont_add_2 = False
                for already in result:
                    if already.issubset(conj):
                        dont_add_2 = True
                        break
                if not dont_add_2:
                    result.append(conj)
    return result











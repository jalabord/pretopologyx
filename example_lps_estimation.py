import numpy as np
import time
import itertools
from pretopologyx.space.pretopological_space_grid import PretopologicalSpaceGrid
from pretopologyx.structure.closures import closures, minimal_closed_sets
import pretopologyx.lsp.biggest_lower_dnf as bldnf


def lps_estimation(size=20, original_dnf=list(), percent_blocked=0, percent_closures=30):

    pre_space = PretopologicalSpaceGrid(np.zeros((size, size)), [[i] for i in range(8)], original_dnf)
    pre_space.block_environment(percent_blocked)
    pre_space.block_neighborhoods()

    initial_set = select_initial_set(pre_space.env, int(round(pre_space.size * percent_closures / 100)))
    list_closures = closures(pre_space, [initial_set])

    start = time.time()
    conj_forb = bldnf.conjonctions_forbidden(pre_space, list_closures)
    estimated_dnf = bldnf.estimate_dnf(conj_forb, len(pre_space.network_index))
    stop = time.time()
    estimation_time = stop - start

    # We make sure the estimation has the same results. We should be able to only change the dnf
    # estimated_pre_space = PretopologicalSpaceGrid(pre_space.env, [[i] for i in range(8)], estimated_dnf)
    # estimated_closures = closures(estimated_pre_space, [initial_set])
    # print(compare_closures(list_closures, estimated_closures))

    return {"dnf": estimated_dnf, "time": estimation_time}


def select_initial_set(env, size):
    options = np.argwhere(env == 0)
    idx = options[np.random.choice(len(options), size, replace=False)]
    id_seeds = idx[:, 0] * len(env) + idx[:, 1]
    initial_set = np.zeros(len(env) ** 2)
    initial_set[id_seeds] = 1
    return initial_set


# We compare the original with the estimated closures to be sure
def compare_closures(real_closures, estimated_closures):
    equal = True
    for i in range(len(real_closures)):
        if not np.array_equal(real_closures[i], estimated_closures[i]):
            equal = False
            break
    return equal


def transform_numbers_to_sets(conj_forbidden, number_neighs):
    fmt = '0' + str(number_neighs) + 'b'
    forbidden_conjs = list()
    for conj in conj_forbidden:
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
    return forbidden_conjs


if __name__ == '__main__':
    sizes = [15, 25, 35]
    DNFs = [
        [{3}, {5}, {6}],
        [{3, 5}, {4, 7}, {6}],
        [{2}, {4}, {1, 3}, {3, 6}, {5, 6, 7}]
    ]
    percentages_blocked = [0, 10, 20, 30, 40, 50, 60]
    print(lps_estimation(20, [{3}, {5}, {6}], 50, 30))

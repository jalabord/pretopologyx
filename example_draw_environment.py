import numpy as np
from pretopologyx.space.pretopological_space_grid import PretopologicalSpaceGrid
from pretopologyx.space.pretopological_space_grid import draw_environment, draw_closures
from pretopologyx.structure.closures import closures, minimal_closed_sets


def draw_grid(size=20, dnf=list(), percent_blocked=30, percent_closures=30):

    pre_space = PretopologicalSpaceGrid(np.zeros((size, size)), [[i] for i in range(8)], dnf)
    pre_space.block_environment(percent_blocked)
    pre_space.block_neighborhoods()

    initial_set = select_initial_set(pre_space.env, int(round(pre_space.size * percent_closures / 100)))
    list_closures = closures(pre_space, [initial_set])

    # We draw a closure
    closure_index = 35
    pre_space.modify_environment(list_closures[closure_index], 1)
    i_set = np.zeros(pre_space.size)
    i_set[np.argwhere(initial_set == 1)[closure_index, 0]] = 1
    pre_space.modify_environment(i_set, 2)
    draw_environment(pre_space.env)

    # for i in range(50):
    #    draw_closures(3, list_closures, pre_space, initial_set, ("draw" + str(i)), dy=0)


def select_initial_set(env, size):
    options = np.argwhere(env == 0)
    idx = options[np.random.choice(len(options), size, replace=False)]
    id_seeds = idx[:, 0] * len(env) + idx[:, 1]
    initial_set = np.zeros(len(env) ** 2)
    initial_set[id_seeds] = 1
    return initial_set


if __name__ == '__main__':
    sizes = [15, 25, 35]
    DNFs = [
        [{3}, {5}, {6}],
        [{3, 5}, {4, 7}, {6}],
        [{2}, {4}, {1, 3}, {3, 6}, {5, 6, 7}]
    ]
    percentages_blocked = [0, 10, 20, 30, 40, 50, 60]
    print(draw_grid(20, [{3}, {5}, {6}], 50, 30))


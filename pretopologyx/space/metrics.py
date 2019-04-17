# Here we define the class in charge of stocking the space and the method for the pseudoclure
# We need to be more consistent about when the copy of a parameter is made
import numpy as np
import cvxpy as cvx
from pretopologyx.space.pretopological_space import PretopologicalSpace, Prenetwork


# pseudoclosure for a pretopologicalspace defined with families
def pseudoclosure(pre_space, initial_set):
    # We are calculating more than once the pseudoclosure, we need to stock it
    disj_list_pseudocl = list()
    for conj in pre_space.dnf:
        conj_list_pseudocl = list()
        for neigh_ind in conj:
            inf_set_t = np.copy(initial_set)
            neigh = pre_space.prenetworks[pre_space.network_index[neigh_ind]].network
            # Here is where need to adapt in order not to recalculate everything for the networks of a same family
            inf_set_t[np.dot(neigh, inf_set_t) >= pre_space.thresholds[neigh_ind]] = 1
            conj_list_pseudocl.append(inf_set_t)
        conj_pseudocl = np.logical_and.reduce(conj_list_pseudocl)
        disj_list_pseudocl.append(conj_pseudocl)
    return np.logical_or.reduce(disj_list_pseudocl).astype('float')


# pseudoclosure for a pretopologicalspace defined with families
def pseudoclosure_prenetwork(pre_space, initial_set):
    # We regroup all networks of the same prenetwork
    networks = [net for conj in pre_space.dnf for net in conj]
    unique_networks = np.unique(networks)
    used_networks = np.zeros(len(pre_space.network_index))
    used_networks[unique_networks] = 1

    for i, prenet in enumerate(pre_space.prenetworks):
        prenetwork_used = np.zeros(len(pre_space.network_index))
        prenetwork_used[pre_space.network_index == i] = 1
        threshold_indices = prenetwork_used*used_networks

        inf_set_t = np.copy(initial_set)
        strength_to_set = np.dot(prenet.network, inf_set_t)

        # We need to include the weights now !!
        # We will add couples of weight threshold
        for th in pre_space.threholds[threshold_indices.astype('bool')]:
            inf_set_t[strength_to_set >= th] = 1
    # We are calculating more than once the pseudoclosure, we need to stock it
    disj_list_pseudocl = list()
    for conj in pre_space.dnf:
        conj_list_pseudocl = list()
        for neigh_ind in conj:
            inf_set_t = np.copy(initial_set)
            neigh = pre_space.prenetworks[pre_space.network_index[neigh_ind]].network
            # Here is where need to adapt in order not to recalculate everything for the networks of a same family
            inf_set_t[np.dot(neigh, inf_set_t) >= pre_space.thresholds[neigh_ind]] = 1
            conj_list_pseudocl.append(inf_set_t)
        conj_pseudocl = np.logical_and.reduce(conj_list_pseudocl)
        disj_list_pseudocl.append(conj_pseudocl)
    return np.logical_or.reduce(disj_list_pseudocl).astype('float')


# node A, on graph with N nodes
# Sum over each neighbor B of A, of 2^(N - |neigh(B) + 1|): That's the number
# sets that don't include B in its pseudo closure, but would include if we
# add A to the group.
# done, we need to check
def teambuilder(pre_space):
    # It only works with one network for the moment
    network = pre_space.neighborhoods[0]
    result = {}
    for el in range(pre_space.size):
        teambuilder_index = [(pre_space.size - sum(network[neigh, :])) for neigh in np.argwhere(network[el, :] != 0)]
        teambuilder_index.sort(reverse=True)
        result[el] = teambuilder_index
    return result


# done, we need to check
def best_pseudoclosure(pre_space, method, *args):
    function_name = "best_" + method
    result = globals()[function_name](*args)
    return result


# We return the node that increases the most the size of the team
# done, we need to check
def best_greedy_teambuilder(pre_space, initial_team, team_size):
    best_teambuilder = 0
    best_teambuilder_gain = 0
    size_team_pseudoclosure = sum(pseudoclosure(pre_space, initial_team))
    for n in range(pre_space.size):
        if not initial_team[n]:
            temporal_team = initial_team.copy()
            temporal_team[n] = 1
            temporal_team_gain = sum(pseudoclosure(pre_space, temporal_team)) - size_team_pseudoclosure
        if temporal_team_gain > best_teambuilder_gain:
            best_teambuilder = temporal_team
            best_teambuilder_gain = temporal_team_gain
    if sum(best_teambuilder) == team_size:
        return best_teambuilder
    else:
        return best_greedy_teambuilder(pre_space, best_teambuilder, team_size)


# One network for the moment
# done, we need to check
def best_ilp(pre_space, team_size, solver="ECOS_BB"):
    adj = pre_space.prenetworks[0].network
    threshold = pre_space.prenetworks[0].thresholds[0]
    diag = np.zeros((pre_space.size, pre_space.size))
    sum_negs = adj.copy()
    sum_negs[adj >= 0] = 0
    np.fill_diagonal(diag, abs(np.sum(sum_negs, axis=1)) + threshold)
    final = np.hstack([adj, diag])
    final_line = np.hstack([np.ones(pre_space.size), np.zeros(pre_space.size)])
    b = np.repeat(threshold, pre_space.size)
    x = cvx.Variable(2*pre_space.size, boolean=True)
    constraints = [final*x >= b, final_line*x == np.array(team_size)]
    obj = cvx.Minimize(np.concatenate((np.zeros(pre_space.size), np.ones(pre_space.size)))*x)
    prob = cvx.Problem(obj, constraints)
    prob.solve(solver=solver)
    if x.value is not None:
        team = (x.value > 0.9)[:pre_space.size].astype('int')
        pseudoclosure_team = np.invert((x.value > 0.9)[pre_space.size:]).astype('int')
        return [team, pseudoclosure_team, final, final_line, b]
    else:
        return [prob, 'problem']


# done, we need to check
def best_random_search(pre_space, set_size, sample_size):
    best_option = list()
    biggest_pseudoclosure = 0
    for i in range(sample_size):
        temporal_targets = np.random.choice(pre_space.size, set_size, replace=False)
        temporal_pseudoclosure = pseudoclosure(pre_space)
        if temporal_pseudoclosure > biggest_pseudoclosure:
            biggest_pseudoclosure = temporal_pseudoclosure
            best_option = np.copy(temporal_targets)
    return best_option


def best_sim_annealing():
    pass


def best_genetic():
    pass


if __name__ == "__main__":
    a = np.random.choice(np.arange(-50, 51), 2500).reshape(50, 50) / 10
    prenet = Prenetwork(a, [2])
    pre_space = PretopologicalSpace([prenet], [0])
    ans = best_ilp(pre_space, 4)
    print("done")
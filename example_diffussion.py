#!/usr/bin/env python
import time
import networkx as nx
import random, math
import numpy as np
import copy
import matplotlib.pyplot as plt
from operator import itemgetter
import pretopologyx.space.metrics as metrics
from pretopologyx.space.pretopological_space_env import PretopologicalSpaceEnv, Prenetwork
from pretopologyx.netgenerator.homophilic import homophilic_network
from pretopologyx.netgenerator.contradictory import contradictory_network


# 'constants'
NORMAL_MODEL = 0
MIXED_MODEL = 1

STATUS = 0
MOTIVATION = 1
CONFORMITY = 2
THRESHOLD = 3
OPINION = 4
OPINION_NEXT = 5

DIFF_CASCADE = 0
DIFF_THRESHOLD = 1
DIFF_UTILITY = 2
DIFF_PRETOPOLOGY = 3

# rnd       0: random,
# det       1: largest degree,
# det       2: smallest degree,
# det       3: largest conformity,
# det       4: smallest conformity,
# det       5: largest motivation,
# det       6: smallest motivation,
# det       7: largest team_builder,
# det       8: smallest team_builder,
# rnd       9: team_builder_for_team
# det       10: eigenvector_centrality,
# det       11: pagerank_centrality,
# rnd       12: random pseudoclosure
# rnd       13: team_builder_for_team_mm
# rnd       14: random pseudoclosure mixed model
# det       15: betweenness_centrality,
# det       16: closeness_centrality,


def diffusion():
    # META PARAMETER
    number_of_populations = 15
    model = 0       # 0 : normal, 1 : mixed model

    # parameters
    pop_size = 1000
    homophily_levels = 6
    random_interventions = [0, 9, 12, 13, 14]

    # parameters for homophilic network
    average_connectivity = 10

    # results
    population_list = list()
    homophily_type_results = {}
    homophily_type_pseudoclosures = {}
    homophily_level_results = {}
    homophily_level_pseudoclosures = {}

    homophily_types = ["status", ] if model == MIXED_MODEL else ["status", "motivation", "conformity"]

    for pop_index in range(number_of_populations):

        population = create_agents(pop_size)
        for homophily_type in homophily_types:
            print("We start homophily type: " + homophily_type)

            for homophily_level in np.linspace(0, 1, homophily_levels):
                homophily_level = round(homophily_level, 2)
                print("     We start homophily level: " + str(homophily_level))

                homo_net = homophilic_network(population, ["status", "motivation", "conformity", "threshold"],
                                              homophily_type, homophily_level, average_connectivity, 0.23)
                prenet = Prenetwork(homo_net, [1])
                pre_space = PretopologicalSpaceEnv([prenet], [[0]], population,
                                                   ["status", "motivation", "conformity", "threshold", "opinion",
                                                    "opinion_next"])
                homophily_G = nx.from_numpy_array(homo_net.T)

                if model == MIXED_MODEL:
                    interventions_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                    enemy_net = contradictory_network(homophilic_network, [homo_net], pop_size)
                    enemy_prenet = Prenetwork(enemy_net, [1])
                    pre_space.add_prenetworks([enemy_prenet])
                else:
                    interventions_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16]

                network_results = start_simulation(pre_space, model, interventions_list, random_interventions, homophily_G)

                homophily_level_results[str(homophily_level)] = copy.deepcopy(network_results[0])
                homophily_level_pseudoclosures[str(homophily_level)] = copy.deepcopy(network_results[1])

            homophily_type_results[str(homophily_type)] = copy.deepcopy(homophily_level_results)
            homophily_type_pseudoclosures[str(homophily_type)] = copy.deepcopy(homophily_level_pseudoclosures)

        population_results = {
            "propagation": copy.deepcopy(homophily_type_results),
            "pseudoclosures": copy.deepcopy(homophily_type_pseudoclosures)
        }
        population_list.append(population_results)
    show_final_matrices()


def start_simulation(pre_space, model, interventions_list, random_interventions, homophily_G):

    # simulation parameters
    # 0: cascade model, 1: threshold model, 2: utility model, 3: pretopological
    diff_models = [0, 1, 2] if model == NORMAL_MODEL else [3, ]
    targets_size = 50
    runs_random = 15
    discussion_runs = 20

    # results
    network_results = list()
    network_pseudoclosures = list()

    for counter_intervention, intervention in enumerate(interventions_list):  # target selection type

        pseudoclosures = list()
        intervention_results = list()
        simulation_runs = runs_random if intervention in random_interventions else 1
        for sim_run in range(simulation_runs):

            targets = select_targets(pre_space, intervention, targets_size, homophily_G)
            simulation_results = list()
            for diff_model in diff_models:

                pre_space.env[:, OPINION] = np.copy(targets)
                pre_space.env[:, OPINION_NEXT] = np.copy(targets)

                newly_converted = np.where(pre_space.env[:, OPINION] != 0)[0]  # used on cascade
                model_results = list()
                for disc_run in range(discussion_runs):

                    model_results.append(np.copy(agents_np[:, OPINION]))

                    if diff_model == DIFF_CASCADE:
                        diffusion_cascade(newly_converted)
                        converted_temp = pre_space.env[:, OPINION_NEXT] - pre_space.env[:, OPINION]
                        newly_converted = np.where(converted_temp != 0)[0]
                    elif diff_model == DIFF_THRESHOLD:
                        diffusion_threshold()
                    elif diff_model == DIFF_UTILITY:
                        diffusion_utility()
                    elif diff_model == DIFF_PRETOPOLOGY:
                        diffusion_pretopology()

                    pre_space.env[:, OPINION] = np.copy(pre_space.env[:, OPINION_NEXT])

                simulation_results.append(np.array(model_results).transpose())
            intervention_results.append(np.array(simulation_results))
            pseudoclosures.append(metrics.pseudoclosure(targets))
        network_results.append(np.array(intervention_results))
        network_pseudoclosures.append(np.array(pseudoclosures))

    return [network_results, network_pseudoclosures]


def create_agents(pop_size):

    status_array = np.random.random(pop_size)
    motivation_array = 2*np.random.random(pop_size)-1
    conformity_array = np.random.random(pop_size)
    threshold_array = np.random.random(pop_size)
    opinion_present_t = np.repeat(0, pop_size)
    opinion_next_t = np.repeat(0, pop_size)
    population = np.column_stack((status_array, motivation_array, conformity_array, threshold_array, opinion_present_t,
                                  opinion_next_t))
    return population


# diffusion is from 0 -> 1
def diffusion_cascade(pre_space, newly_converted):
    global agents_np

    for p in newly_converted:
        for n in G.neighbors(p):
            if agents_np[int(n), int(OPINION_T)] == 0:
                rand = random.uniform(0, 1)
                if rand < G[int(p)][int(n)]["influence"]:
                    agents_np[int(n), int(OPINION_NEXT_T)] = 1


# diffusion is from 0 -> 1
def diffusion_threshold(pre_space):
    global agents_np

    for p in range(pop_size):
        if agents_np[int(p), OPINION_T] == 0:
            if calculate_neighbors_one_percentage(p) > agents_np[int(p), int(THRESHOLD)]:

                agents_np[int(p), OPINION_NEXT_T] = 1


# diffusion is from 0 -> 1
def diffusion_utility(pre_space):
    global agents_np

    for p in range(pop_size):
        # U_zero = conformity*neigh_zero + (1 - conformity)*indiv_U_zero
        # U_one = conformity*neigh_one + (1 - conformity)*indiv_U_one
        # U_change = conformity*(neigh_one - neigh_zero) + (1 - conformity)*motivation > 0
        # motivation = indiv_U_one - indiv_U_zero
        if agents_np[int(p), int(OPINION_T)] == 0:
            if agents_np[int(p), int(CONFORMITY)]*(2*calculate_neighbors_one_percentage(p) - 1) + \
             (1 - agents_np[int(p), int(CONFORMITY)])*agents_np[int(p), int(MOTIVATION)] > 0:

                agents_np[int(p), int(OPINION_NEXT_T)] = 1


# diffusion is from 0 -> 1
# enemies decrease by one the friends
def diffusion_pretopology(pre_space):
    global agents_np

    for p in range(pop_size):
        if agents_np[int(p), OPINION_T] == 0:
            if calculate_positive_neighbors_one_percentage(p) > agents_np[int(p), int(THRESHOLD)]:

                agents_np[int(p), OPINION_NEXT_T] = 1


def select_targets(pre_space, intervention, targets_size, homophily_G):
    # ordered by degree
    targets = np.zeros(pre_space.size)
    potential_targets = list()
    if intervention == 0:     # 0: random,
        potential_targets = np.random.choice(pre_space.size, targets_size, replace=False)
    if intervention == 1:     # 1: largest degree,
        potential_targets = [x[0] for x in sorted(homophily_G.degree(), key=itemgetter(1), reverse=True)]
    elif intervention == 2:       # 2: smallest degree,
        potential_targets = [x[0] for x in sorted(homophily_G.degree(), key=itemgetter(1))]

    elif intervention == 3:       # 3: largest conformity,
        potential_targets = agents_np[:, CONFORMITY].argsort()[::-1]

    elif intervention == 4:       # 4: smallest conformity,
        potential_targets = agents_np[:, CONFORMITY].argsort()

    elif intervention == 5:       # 5: largest motivation,
        potential_targets = agents_np[:, MOTIVATION].argsort()[::-1]

    elif intervention == 6:       # 6: smallest motivation,
        potential_targets = agents_np[:, MOTIVATION].argsort()

    elif intervention == 7:       # 7: largest team_builder,
        potential_targets = [x[0] for x in sorted(metrics.teambuilder(pre_space).items(), key=itemgetter(1),
                                                  reverse=True)]

    elif intervention == 8:       # 8: smallest team_builder,
        potential_targets = [x[0] for x in sorted(metrics.teambuilder(pre_space).items(), key=itemgetter(1))]

    elif intervention == 9:       # 9: team_builder_for_team
        potential_targets = metrics.best_greedy_teambuilder(pre_space, np.zeros(pre_space.size), targets_size)

    elif intervention == 10:      # 10: eigenvector_centrality,
        potential_targets = [x[0] for x in sorted(nx.eigenvector_centrality_numpy(homophily_G).items(),
                                                  key=itemgetter(1), reverse=True)]

    elif intervention == 11:      # 11: pagerank_centrality,
        potential_targets = [x[0] for x in sorted(nx.pagerank(homophily_G).items(), key=itemgetter(1), reverse=True)]

    elif intervention == 12:      # 12: random pseudoclosure
        potential_targets = metrics.best_random_search(pre_space, targets_size, 500)

    elif intervention == 13:      # 13: team_builder_for_team_mm
        potential_targets = metrics.best_greedy_teambuilder(pre_space, np.zeros(pre_space.size), targets_size)

    elif intervention == 14:      # 14: random pseudoclosure mixed model
        potential_targets = metrics.best_random_search(pre_space, targets_size, 500)

    elif intervention == 15:      # 15: betweenness_centrality,
        potential_targets = [x[0] for x in sorted(nx.betweenness_centrality(homophily_G).items(), key=itemgetter(1),
                                                  reverse=True)]

    elif intervention == 16:      # 16: closeness_centrality,
        potential_targets = [x[0] for x in sorted(nx.closeness_centrality(homophily_G).items(), key=itemgetter(1),
                                                  reverse=True)]

    targets[np.array(potential_targets)[:targets_size]] = 1
    return targets

    # Test for similarity with other measures


def calculate_neighbors_one_percentage(pre_space, node):
    neighbors_opinion = 0
    adj = pre_space.prenetworks[0].network[0]
    neighbors = (adj[node, :] != 0)
    neighbors_opinion = np.dot(pre_space.env[:, OPINION][neighbors])
    neighbors_number = sum(neighbors)
    return neighbors_opinion/neighbors_number if neighbors_number != 0 else 0


def calculate_positive_neighbors_one_percentage(pre_space, node):
    neighbors_opinion = 0
    for n in G.neighbors(node):
        neighbors_opinion += agents_np[int(n), int(OPINION_T)]
    for n in G_enemy.neighbors(node):
        neighbors_opinion -= agents_np[int(n), int(OPINION_T)]
    if len(list(G.neighbors(node))):
        return neighbors_opinion / len(list(G.neighbors(node)))
    else:
        return 0


# For analysis

def create_summary(homophily_types, homophily_levels, interventions_list):

    summary = {}
    summary_std = {}
    summary_pseudoclosures = {}
    summary_pseudoclosures_std = {}

    for i1 in range(len(homophily_types)):
        for i2 in np.linspace(0, 1, homophily_levels):
            if mixed_model:
                for i3 in range(len(interventions_list)):
                    for i4 in range(len(networks_list)):
                        name = "expe" + str(i1) + "-" + str(round(i2, 2)) + "-" + str(i3)

                        print(name + "       " + str(i4))

                        if i4 == 0:
                            stack_temp = \
                                np.array(networks_list[i4]["propagation"][str(i1)][str(round(i2, 2))][str(i3)])
                            stack_pseudoclosure_temp = \
                                np.array(networks_list[i4]["pseudoclosures"][str(i1)][str(round(i2, 2))][str(i3)])
                        else:
                            stack_temp = \
                                np.concatenate((stack_temp,
                                                networks_list[i4]["propagation"][str(i1)][str(round(i2, 2))][str(i3)]))
                            stack_pseudoclosure_temp = \
                                np.concatenate((stack_pseudoclosure_temp,
                                                networks_list[i4]["pseudoclosures"][str(i1)][str(round(i2, 2))][str(i3)]))

                    summary[name] = stack_temp.sum(axis=1).mean(axis=0)
                    summary_std[name] = stack_temp.sum(axis=1).std(axis=0)
                    summary_pseudoclosures[name] = stack_pseudoclosure_temp.mean(axis=0)
                    summary_pseudoclosures_std[name] = stack_pseudoclosure_temp.std(axis=0)

            else:
                for i3 in range(3):     # model type
                    for i4 in range(len(interventions_list)):
                        for i5 in range(len(networks_list)):
                            name = "expe" + str(i1) + "-" + str(round(i2, 2)) + "-" + str(i3) + str(i4)

                            if i5 == 0:
                                stack_temp = \
                                    np.array(networks_list[i5]["propagation"]
                                             [str(i1)][str(round(i2, 2))][str(i3)][str(i4)])
                                stack_pseudoclosure_temp = \
                                    np.array(networks_list[i5]["pseudoclosures"]
                                             [str(i1)][str(round(i2, 2))][str(i3)][str(i4)])
                            else:
                                stack_temp = \
                                    np.concatenate((stack_temp,
                                                    networks_list[i5]["propagation"]
                                                    [str(i1)][str(round(i2, 2))][str(i3)][str(i4)]))
                                stack_pseudoclosure_temp = \
                                    np.concatenate((stack_pseudoclosure_temp,
                                                    networks_list[i5]["pseudoclosures"]
                                                    [str(i1)][str(round(i2, 2))][str(i3)][str(i4)]))
                        summary[name] = stack_temp.sum(axis=1).mean(axis=0)
                        summary_std[name] = stack_temp.sum(axis=1).std(axis=0)
                        summary_pseudoclosures[name] = stack_pseudoclosure_temp.mean(axis=0)
                        summary_pseudoclosures_std[name] = stack_pseudoclosure_temp.std(axis=0)


def create_final_matrix(homophily_types, homophily_levels, interventions_list, discussion_runs):

    final_matrix = list()
    final_pseudoclosures_matrix = list()

    for i1 in range(len(homophily_types)):   # homophily types
        for i2 in np.linspace(0, 1, homophily_levels):
            if mixed_model:
                for i3 in range(len(interventions_list)):
                    name = "expe" + str(i1) + "-" + str(round(i2, 2)) + "-" + str(i3)

                    row = list()
                    row.append(i1)
                    row.append(round(i2, 2))
                    row.append(i3)
                    row.append(summary[name][discussion_runs-1])
                    row.append(summary_std[name][discussion_runs - 1])
                    final_matrix.append(row)

                    row_pseudoclosure = list()
                    row_pseudoclosure.append(i1)
                    row_pseudoclosure.append(round(i2, 2))
                    row_pseudoclosure.append(i3)
                    row_pseudoclosure.append(summary_pseudoclosures[name][0])
                    row_pseudoclosure.append(summary_pseudoclosures_std[name][0])
                    row_pseudoclosure.append(summary_pseudoclosures[name][1])
                    row_pseudoclosure.append(summary_pseudoclosures_std[name][1])
                    final_pseudoclosures_matrix.append(row_pseudoclosure)
            else:
                for i3 in range(3):
                    for i4 in range(len(interventions_list)):
                        name = "expe" + str(i1) + "-" + str(round(i2, 2)) + "-" + str(i3) + str(i4)

                        row = list()
                        row.append(i1)
                        row.append(round(i2, 2))
                        row.append(i3)
                        row.append(i4)
                        row.append(summary[name][discussion_runs-1])
                        row.append(summary_std[name][discussion_runs - 1])
                        final_matrix.append(row)

                        row_pseudoclosure = list()
                        row_pseudoclosure.append(i1)
                        row_pseudoclosure.append(round(i2, 2))
                        row_pseudoclosure.append(i3)
                        row_pseudoclosure.append(i4)
                        row_pseudoclosure.append(summary_pseudoclosures[name][0])
                        row_pseudoclosure.append(summary_pseudoclosures_std[name][0])
                        row_pseudoclosure.append(summary_pseudoclosures[name][1])
                        row_pseudoclosure.append(summary_pseudoclosures_std[name][1])
                        final_pseudoclosures_matrix.append(row_pseudoclosure)

    final_matrix = np.array(final_matrix)
    final_pseudoclosures_matrix = np.array(final_pseudoclosures_matrix)


def show_final_matrices(homophily_types, homophily_levels):

    create_summary()
    create_final_matrix()

    for i1 in range(len(homophily_types)):  # number of homophily types
        for i2 in np.linspace(0, 1, homophily_levels):
            if mixed_model:
                print("MATRIX: " + str(i1) + "-" + str(round(i2, 2)))
                print(final_matrix[np.logical_and.reduce(np.array([final_matrix[:, 0] == i1,
                                                                   final_matrix[:, 1] == round(i2, 2)]))])
            else:
                for i3 in range(3):
                    print("MATRIX: " + str(i1) + "-" + str(round(i2, 2)) + "-" + str(i3))
                    print(final_matrix[np.logical_and.reduce(np.array([final_matrix[:, 0] == i1,
                                                                       final_matrix[:, 1] == round(i2, 2),
                                                                       final_matrix[:, 2] == i3]))])

    plot_results(homophily_types, homophily_levels)


def mean_pseudoclosures(homophily_types, homophily_levels, interventions_list):

    for i1 in range(len(homophily_types)):  # number of homophily types
        for i2 in np.linspace(0, 1, homophily_levels):
            if mixed_model:
                for i3 in range(len(interventions_list)):
                    print("MATRIX: " + str(i1) + "-" + str(round(i2, 2)) + "-" + str(i3))
                    print(fixed_homophily_type_pseudoclosures[str(i1)][str(i2)][str(i3)].mean(axis=0))
            else:
                for i3 in range(3):  # model_type
                    for i4 in range(len(interventions_list)):
                        print("MATRIX: " + str(i1) + "-" + str(round(i2, 2)) + "-" + str(i3))
                        print(fixed_homophily_type_pseudoclosures[str(i1)][str(i2)][str(i3)][str(i4)].mean(axis=0))

    for i in range(16):
        fixed_homophily_type_pseudoclosures["2"]["1.0"][str(i)].mean(axis=0)


def plot_results(final_matrix, final_pseudoclosures_matrix, homophily_types, homophily_levels, mixed_model):

    if mixed_model:
        col = ['#D32F2F', '#FF4081', '#7B1FA2', '#7C4DFF', '#303F9F', '#1976D2', '#03A9F4', '#00BCD4', '#009688',
               '#4CAF50', '#8BC34A', '#CDDC39', '#FFEB3B', '#FFC107', '#FF9800', '#FF5722', '#795548']
        labels = ["random", "largest degree", "smallest degree", "largest conformity", "smallest conformity",
                  "largest motivation", "smallest motivation", "largest team_builder", "smallest team_builder",
                  "team_builder_for_team", "eigenvector_centrality", "pagerank_centrality", "random pseudoclosure",
                  "team_builder_for_team_mm", "random pseudoclosure mixed model", "betweenness_centrality",
                  "closeness_centrality"]
        for i1 in range(len(homophily_types)):  # homophily type

            # Propagation plot
            fig, axes = plt.subplots(3, 2)

            for index, i2 in enumerate(np.linspace(0, 1, homophily_levels)):
                table_name = "HT: " + str(i1) + " - Level: " + str(round(i2, 2))
                data = final_matrix[np.logical_and.reduce(np.array([final_matrix[:, 0] == i1,
                                                                    final_matrix[:, 1] == round(i2, 2)]))]
                rects = axes[math.floor(index / 2), (index % 2)]\
                    .bar(data[:, 2], data[:, 3], yerr=data[:, 4], color=col, align='center', alpha=0.5)
                axes[math.floor(index / 2), (index % 2)].set_ylabel('Final Opinion B')
                axes[math.floor(index / 2), (index % 2)].set_title(table_name)

                # We only plot the legend once
                if i1 == 0 and index == 0:
                    axes[math.floor(index / 2), (index % 2)].legend(rects, labels)

            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig("MM_homophily_type_" + str(i1) + ".svg", format="svg")

            plt.close('all')

            # Normal pseudoclosure plot
            fig, axes = plt.subplots(3, 2)

            for index, i2 in enumerate(np.linspace(0, 1, homophily_levels)):
                table_name = "HT: " + str(i1) + " - Level: " + str(round(i2, 2))
                data = \
                    final_pseudoclosures_matrix[np.logical_and.reduce
                                                (np.array([final_pseudoclosures_matrix[:, 0] == i1,
                                                           final_pseudoclosures_matrix[:, 1] == round(i2, 2)]))]
                rects = axes[math.floor(index / 2), (index % 2)]\
                    .bar(data[:, 2], data[:, 3], yerr=data[:, 4], color=col, align='center', alpha=0.5)
                axes[math.floor(index / 2), (index % 2)].set_ylabel('Pseudoclosure size')
                axes[math.floor(index / 2), (index % 2)].set_title(table_name)

                # We only plot the legend once
                if i1 == 0 and index == 0:
                    axes[math.floor(index / 2), (index % 2)].legend(rects, labels)

            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig("Normal Pseudoclosures_MM_homophily_type_" + str(i1) + ".svg", format="svg")

            plt.close('all')

            # Mixed model pseudoclosure plot
            fig, axes = plt.subplots(3, 2)

            for index, i2 in enumerate(np.linspace(0, 1, homophily_levels)):
                table_name = "HT: " + str(i1) + " - Level: " + str(round(i2, 2))
                data = \
                    final_pseudoclosures_matrix[np.logical_and.reduce
                                                (np.array([final_pseudoclosures_matrix[:, 0] == i1,
                                                           final_pseudoclosures_matrix[:, 1] == round(i2, 2)]))]
                rects = axes[math.floor(index / 2), (index % 2)]\
                    .bar(data[:, 2], data[:, 5], yerr=data[:, 6], color=col, align='center', alpha=0.5)
                axes[math.floor(index / 2), (index % 2)].set_ylabel('Pseudoclosure size')
                axes[math.floor(index / 2), (index % 2)].set_title(table_name)

                # We only plot the legend once
                if i1 == 0 and index == 0:
                    axes[math.floor(index / 2), (index % 2)].legend(rects, labels)

            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.savefig("MM_Pseudoclosures_MM_homophily_type_" + str(i1) + ".svg", format="svg")
    else:
        col = ['#D32F2F', '#FF4081', '#7B1FA2', '#7C4DFF', '#303F9F', '#1976D2', '#03A9F4', '#00BCD4', '#009688',
               '#4CAF50', '#8BC34A', '#CDDC39', '#FFEB3B', '#FFC107', '#FF9800']
        labels = ["random", "largest degree", "smallest degree", "largest conformity", "smallest conformity",
                  "largest motivation", "smallest motivation", "largest team_builder", "smallest team_builder",
                  "team_builder_for_team", "eigenvector_centrality", "pagerank_centrality", "random pseudoclosure",
                  "betweenness_centrality", "closeness_centrality"]
        for i3 in range(3):  # model type
            for i1 in range(len(homophily_types)):  # homophily type

                # Propagation plot
                fig, axes = plt.subplots(3, 2)

                for index, i2 in enumerate(np.linspace(0, 1, homophily_levels)):
                    table_name = "HT: " + str(i1) + " - Level: " + str(round(i2, 2)) + " - Model: " + str(i3)
                    data = final_matrix[np.logical_and.reduce(np.array([final_matrix[:, 0] == i1,
                                                                        final_matrix[:, 1] == round(i2, 2),
                                                                        final_matrix[:, 2] == i3]))]
                    rects = axes[math.floor(index / 2), (index % 2)]\
                        .bar(data[:, 3], data[:, 4], yerr=data[:, 5], color=col, align='center', alpha=0.5)
                    axes[math.floor(index / 2), (index % 2)].set_ylabel('Pseudoclosure size')
                    axes[math.floor(index / 2), (index % 2)].set_title(table_name)

                    # We only plot the legend once
                    if i1 == 0 and index == 0:
                        axes[math.floor(index / 2), (index % 2)].legend(rects, labels)

                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
                plt.savefig("model_type_" + str(i3) + "homophily_type_" + str(i1) + ".svg", format="svg")
                plt.close('all')

                # Normal Pseudoclosure plot
                fig, axes = plt.subplots(3, 2)

                for index, i2 in enumerate(np.linspace(0, 1, homophily_levels)):
                    table_name = "HT: " + str(i1) + " - Level: " + str(round(i2, 2)) + " - Model: " + str(i3)
                    data = final_pseudoclosures_matrix[np.logical_and.reduce
                                                       (np.array([final_pseudoclosures_matrix[:, 0] == i1,
                                                                  final_pseudoclosures_matrix[:, 1] == round(i2, 2),
                                                                  final_pseudoclosures_matrix[:, 2] == i3]))]
                    rects = axes[math.floor(index / 2), (index % 2)]\
                        .bar(data[:, 3], data[:, 4], yerr=data[:, 5], color=col, align='center', alpha=0.5)
                    axes[math.floor(index / 2), (index % 2)].set_ylabel('Final Opinion B')
                    axes[math.floor(index / 2), (index % 2)].set_title(table_name)

                    # We only plot the legend once
                    if i1 == 0 and index == 0:
                        axes[math.floor(index / 2), (index % 2)].legend(rects, labels)

                plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
                plt.savefig("model_type_" + str(i3) + "homophily_type_" + str(i1) + ".svg", format="svg")
                plt.close('all')


if __name__ == '__main__':
    diffusion()

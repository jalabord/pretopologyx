import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import time
import hdbscan
import math
from pretopologyx.space.pretopological_space import PretopologicalSpace, Prenetwork
from pretopologyx.structure.closures import elementary_closures_shortest_degree, elementary_closures_shortest_degree_largeron
import pretopologyx.structure.hierarchy as hierarchy
from pretopologyx.netgenerator.geometric import prenetwork_closest
import sys

sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}


class Pretopocluster:

    def __init__(self):
        self.labels_ = np.array([])

    def fit(self, data):
        self.labels_ = clustering(data)

    def fit_predict(self, data):
        self.labels_ = clustering(data)
        return self.labels_


class PretopoclusterLargeron:

    def __init__(self):
        self.labels_ = np.array([])

    def fit(self, data):
        self.labels_ = clustering(data, method="largeron")

    def fit_predict(self, data):
        self.labels_ = clustering(data, method="largeron")
        return self.labels_


# Parameters multiple comparison: set 4
# with parameters degree=4, th=1.3 we get many clusters reduced to 2 in the two rings example
def clustering(data, method="mine"):

    prenetwork = prenetwork_closest(data)
    pre_space = PretopologicalSpace([prenetwork], [[0]])

    if method == "mine":
        print("mine")
        closures_list = elementary_closures_shortest_degree(pre_space, 4)
    else:
        print("largeron")
        closures_list = elementary_closures_shortest_degree_largeron(pre_space, 4)
    # print_closures_members(closures_list)
    # compare_closures(closures_list, closures_list_largeron)

    # GRAVITY PSEUDOHIERARCHY
    pseudohierarchy_gravity = hierarchy.pseudohierarchy_gravity(closures_list.copy())
    adj_pseudohierarchy_gravity = hierarchy.pseudohierarchy_filter_threshold(pseudohierarchy_gravity.copy(), 0.50)
    answer_gravity = hierarchy.pseudohierarchy_filter_equivalents(adj_pseudohierarchy_gravity.copy(), closures_list)
    filtered_adj_pseudohierarchy_gravity = answer_gravity[0]
    selected_closures_gravity = answer_gravity[1]
    # We select the closures that are at the end of the directed graph (Those that have only one neighbor, himself)
    final_selected_indices_gravity = np.argwhere(np.sum(filtered_adj_pseudohierarchy_gravity, axis=1) == 1).flatten()
    final_selected_closures_gravity = [x for i, x in enumerate(selected_closures_gravity) if i in final_selected_indices_gravity]

    # print("_______________SELECTED CLOSURES GRAVITY:")
    # print_closures_members(selected_closures_gravity)
    # print("_______________FINAL SELECTED CLOSURES GRAVITY:")
    # print_closures_members(final_selected_closures_gravity)

    labels = np.zeros(len(data))
    for i, closure in enumerate(final_selected_closures_gravity):
        labels += (i+1)*closure
        labels[labels > (i+1)] = (i+1)
    labels = labels - 1
    # !!!!!!!!!!! WE ARE ADDING THIS? MAYBE CAN CAUSE A PROBLEM IN THE OTHER EXAMPLE
    labels = labels.astype('int')

    # PLOT
    # plt.scatter(data.T[0], data.T[1], c='b', **plot_kwds)
    # frame = plt.gca()
    # frame.axes.get_xaxis().set_visible(False)
    # frame.axes.get_yaxis().set_visible(False)
    # plot_clusters_2(data, final_selected_closures_gravity)

    return labels


def draw_arrows(data, adj):
    for p in range(len(data)):
        for n in np.argwhere(adj[p, :]):
            if p != n[0]:
                plt.arrow(data[p, 0], data[p, 1], data[n[0], 0] - data[p, 0], data[n[0], 1] - data[p, 1],
                          width=0.000002, color='red', head_length=0.0, head_width=0.00003)


def plot_clusters(data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    # frame = plt.gca()
    # frame.axes.get_xaxis().set_visible(False)
    # frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
    plt.show()


def plot_clusters_axes(axe, data, algorithm, args, kwds):
    labels = algorithm(*args, **kwds).fit_predict(data)
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    axe.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    # frame = plt.gca()
    axe.get_xaxis().set_visible(False)
    axe.get_yaxis().set_visible(False)
    axe.set_title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=14)
    # axe.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)


def plot_clusters_(data, indices):
    start_time = time.time()
    labels = np.zeros(len(data)).astype(int)
    labels[list(indices)] = 1
    end_time = time.time()
    palette = sns.color_palette('deep', int(np.unique(labels).max()) + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)


def plot_clusters_2(data, clusters):
    start_time = time.time()
    labels = np.zeros(len(data)).astype(int)
    for i, cluster in enumerate(clusters):
        labels[list(np.argwhere(cluster == 1).flatten())] = i + 1
    end_time = time.time()
    palette = sns.color_palette('deep', int(np.unique(labels).max()) + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
    plt.savefig("cluster.svg")
    plt.show()
    return labels


def print_closures_members(list_cls):
    for i, el in enumerate(list_cls):
        print("CLOSURE: " + str(i))
        print(np.argwhere(el == 1).flatten())
        print("")


# We make sure both methods retun the same closures
def compare_closures(mine, largeron):
    mine = np.array(mine)
    for cl in largeron:
        if not any((mine[:] == cl).all(1)):
            print("problem")
            return
    print("The are equal")


if __name__ == '__main__':
    # Paramaters: th: 1.95 and degree 3 gives one less, th 2 gives two less
    # Paramaters: th: 1.95 and degree 4 gives still one less, but has many closures reduced to 5 (6, one really small)
    # Paramaters: th: 2.5 and degree 4 gives the perfect clusters

    clusterable_data = np.load('clusterable_data.npy')

    fig, axes = plt.subplots(2, 1)
    fig.set_size_inches(10, 15)

    # plot_clusters_axes(axes[0], clusterable_data, cluster.KMeans, (), {'n_clusters': 6})
    # plot_clusters_axes(axes[1], clusterable_data, cluster.AgglomerativeClustering, (), {'n_clusters': 6, 'linkage': 'ward'})
    plot_clusters_axes(axes[0], clusterable_data, hdbscan.HDBSCAN, (), {'min_cluster_size': 15})
    plot_clusters_axes(axes[1], clusterable_data, Pretopocluster, (), {})
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    # plt.savefig("KMeans_Agglomerative.svg", format="svg")
    plt.savefig("HDBSCAN_Pretopology.svg", format="svg")

    # clustering()




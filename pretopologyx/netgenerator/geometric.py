from scipy.spatial import distance_matrix
import numpy as np
import math
from pretopologyx.space.pretopological_space import Prenetwork


# returns the adjacency matrix of the elements that are closer than the "average distance",
# normalized by the biggest distance
# c.f. calculate average distance
# points is a list of tuples with the two coordinates
def network_ball(dm, points, radius):
    adj = (dm < radius).astype('float')
    # With this parametrisation works fine for the 6 clusters example, besides a lonely small cluster
    # adj = (dm < 1.12*th).astype('float')
    np.fill_diagonal(adj, 0)
    return adj

# We need to reorder this
def network_ball_distances(dm, points, radius):
    adj = (dm > radius)
    dm[adj] = 0
    network = dm/radius
    network = 1 - network
    network[adj] = 0
    np.fill_diagonal(network, 0)
    return network


# calculates an approximation of the distance between elements if they were evenly spaced
# uses the augmenting lines in the grid heuristic
def prenetwork_closest(points, distances=True):
    x_dif = np.amax(points[:, 0]) - np.amin(points[:, 0])
    y_dif = np.amax(points[:, 1]) - np.amin(points[:, 1])
    area = x_dif*y_dif
    square_length = math.sqrt(area / len(points))

    dm = distance_matrix(points, points)
    closest = np.sum(np.ma.masked_less(dm, square_length/2).mask.astype('float'), axis=1)
    inverse = (closest-1)/closest
    real_points = len(points) - sum(inverse)

    # Paramaters: th: 2.5 and degree 4 gives the perfect clusters for HBSDCAN parameters
    # with radius = 2*square and th = len(points)/real_points works perfect for the 6 clusters
    # radius = 2*square_length
    # th = len(points)/(1.2*real_points)

    # with radius = 2*square and th = len(points)/(3*real_points) works very well for the others
    # radius = 2*square_length
    # th = len(points)/(3*real_points)

    # with radius = 2.5*square and th = len(points)/(2*real_points) also works very well for the others
    radius = 2.5*square_length
    th = len(points)/(2*real_points)


    print("")
    print("")
    print("AREA: " + str(area))
    print("Number of points: " + str(len(points)))
    print("Square length: " + str(square_length))
    print("REAL_POINTS: " + str(real_points))
    print("RADIUS: " + str(radius))
    print("TH: " + str(th))

    if distances:
        network = network_ball_distances(dm, points, radius)
    else:
        network = network_ball(dm, points, radius)
    return Prenetwork(network, [th])
    # return Prenetwork(network, [2.5])


# calculates an approximation of the distance between elements if they were evenly spaced
# uses the augmenting lines in the grid heuristic
# This became useless since we realized we could alculate the area by point, ant take the sqrt
def network_closest_distances_useless(points):
    x_dif = np.amax(points[:, 0]) - np.amin(points[:, 0])
    y_dif = np.amax(points[:, 1]) - np.amin(points[:, 1])
    rows_answer = 0
    area = x_dif*y_dif
    accuracy = math.inf
    for rows in range(1, len(points)):
        square_length = x_dif/(len(points)/rows)
        area_grid = rows*square_length*x_dif
        area_dif = area - area_grid
        if abs(area_dif) < accuracy:
            accuracy = area_dif
        else:
            rows_answer = rows - 1
            break
    print(rows_answer)
    square_length = x_dif / (len(points) / rows_answer)
    if rows_answer*square_length < y_dif:
        square_length = y_dif/rows_answer
    print("SQUARE LENGTH ESTIMATION: " + str(square_length))
    print("SQUARE LENGTH OBVIOUS ESTIMATION: " + str(math.sqrt(area/len(points))))
    return network_ball_distances(points, 2*square_length)


# calculates an approximation of the distance between elements if they were evenly spaced
def network_closest_old(points):
    x_dif = np.amax(points[:, 0]) - np.amin(points[:, 0])
    y_dif = np.amax(points[:, 1]) - np.amin(points[:, 1])
    x_points = math.floor(math.sqrt(len(points) * y_dif / x_dif))
    th = x_dif / x_points
    print(th)
    # return network_ball(points, th)
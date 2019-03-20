import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pylab
from mpl_toolkits.mplot3d import Axes3D

fun = 'euclidian'

def normalize(points)
    norm = points
    for i in range(len(points[0])):
        dim = [point[i] for point in points]
        norm = map(lambda point: (point[i] * (min(dim) / max(dim))), norm)


def euclidian(X, Y):
    """
        Takes two points in list format and finds
        the euclidian distance between them
    """
    return math.sqrt(sum([pow(X[i] - Y[i], 2) for i in range(len(X))]))

def manhattan(X, Y):
    """
        Takes two points in list format and finds
        the manhattan distance between them
    """
    return abs(sum([X[i] - Y[i] for i in range(len(X))]))


def make_clusters(k, points, centers):
    """
        Separates each data point into the cluster it is closest to
    """
    # Initialize empty list of clusters
    clusts = [[] for i in range(k)]
    for point in points:
        # iterate for each point to find the closest center
        if fun == 'euclidian':
            dmin = euclidian(point, centers[0])
        else:
            dmin = manhattan(point, centers[0])
        x = 1
        for i in range(1, k):
            if fun == 'euclidian':
                dist = euclidian(point, centers[i])
            else:
                dist = manhattan(point, centers[i])
            if(dist < dmin):
                x = i
                dmin = dist
        # add point to its corresponding cluster
        clusts[x].append(point)
    return clusts 


def calculate_means(clusts, points):
    """
        finds the mean coordinates of each cluster
    """
    means = []
    for clust in clusts:
        # separate each cluster's data by dimension instead of by point
        if len(clust) != 0:
            dims = [[point[i] for point in clust] for i in range(len(clust[0]))]
            # add up each dimension's values, divide by the size
            mean = [(sum(dim) / len(clust)) for dim in dims]
        else:
            # if a cluster is empty, set it to a random point
            mean = points[random.randint(0, len(points) - 1)]
        means.append(mean)
    return means


def k_means(k, points):
    """
        Takes a list of points and the number of clusters,
        and iterates through the k means algorithm
    """
    means = random.sample(points, k)
    centers = None
    while centers != means:
        # set centers to old means
        centers = means
        # create list of clusters
        clusts = make_clusters(k, points, centers)
        # set new means
        means = calculate_means(clusts, points)
    return clusts


num_clusters = 4
fig1 = plt.figure(1, figsize=(4,3))
fig2 = plt.figure(2, figsize=(4,3))
ax = Axes3D(fig1, rect=[0, 0, .95, 1])
ax2 = Axes3D(fig2, rect=[0, 0, .95, 1])
X = [[9.3498486,56.7408757,17.0527715677876],
    [9.3501884,56.7406785,17.614840244389],
    [8.5856624,57.0106364,32.0776406065856],
    [8.5851822,57.0099725,28.7124475240045],
    [10.0015925,56.6369145,75.2648335602078],
    [10.0019142,56.6373512,71.556921301853],
    [9.5685446,57.0522153,1.94681426140926],
    [9.5693017,57.05361,2.24977425104904],
    [9.912351,57.0257797,3.66317365511644],
    [10.1787039,56.592082,33.5740971625897],
    [10.1782748,56.592342,33.8217791473094],
    [10.1779273,56.5923715,33.0209795956494]]
labels = [0, 0, 1, 1, 2, 2, 3, 3, 3, 2, 2, 2]
ax.scatter([row[0] for row in X], [row[1] for row in X], [row[2] for row in X], \
           c=[int(i % num_clusters) for i in labels], cmap=pylab.cm.gist_ncar)

clusts = k_means(num_clusters, X)
for clust, i in zip(clusts, range(num_clusters)):
    print(clust)
#    ax2.scatter([row[0] for row in clust], [row[1] for row in clust], [row[2] for row in clust], \
 #               c='r')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Altitude')
ax.dist = 12

ax2.w_xaxis.set_ticklabels([])
ax2.w_yaxis.set_ticklabels([])
ax2.w_zaxis.set_ticklabels([])
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
ax2.set_zlabel('Altitude')
ax2.dist = 12



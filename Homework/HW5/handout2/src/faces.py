"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Famous Faces
"""

# python libraries
import collections

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt

# libraries specific to project
import util
from util import *
from cluster import *
from collections import defaultdict

######################################################################
# helper functions
######################################################################

def build_face_image_points(X, y) :
    """
    Translate images to (labeled) points.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), features (each row is one image)
        y     -- numpy array of shape (n,), targets
    
    Returns
    --------------------
        point -- list of Points, dataset (one point for each image)
    """
    
    n,d = X.shape
    
    images = collections.defaultdict(list) # key = class, val = list of images with this class
    for i in xrange(n) :
        images[y[i]].append(X[i,:])
    
    points = []
    for face in images :
        count = 0
        for im in images[face] :
            points.append(Point(str(face) + '_' + str(count), face, im))
            count += 1

    return points


def plot_clusters(clusters, title, average) :
    """
    Plot clusters along with average points of each cluster.

    Parameters
    --------------------
        clusters -- ClusterSet, clusters to plot
        title    -- string, plot title
        average  -- method of ClusterSet
                    determines how to calculate average of points in cluster
                    allowable: ClusterSet.centroids, ClusterSet.medoids
    """
    
    plt.figure()
    np.random.seed(20)
    label = 0
    colors = {}
    centroids = average(clusters)
    for c in centroids :
        coord = c.attrs
        plt.plot(coord[0],coord[1], 'ok', markersize=12)
    for cluster in clusters.members :
        label += 1
        colors[label] = np.random.rand(3,)
        for point in cluster.points :
            coord = point.attrs
            plt.plot(coord[0], coord[1], 'o', color=colors[label])
    plt.title(title)
    plt.show()


def generate_points_2d(N, seed=1234) :
    """
    Generate toy dataset of 3 clusters each with N points.
    
    Parameters
    --------------------
        N      -- int, number of points to generate per cluster
        seed   -- random seed
    
    Returns
    --------------------
        points -- list of Points, dataset
    """
    np.random.seed(seed)
    
    mu = [[0,0.5], [1,1], [2,0.5]]
    sigma = [[0.1,0.1], [0.25,0.25], [0.15,0.15]]
    
    label = 0
    points = []
    for m,s in zip(mu, sigma) :
        label += 1
        for i in xrange(N) :
            x = util.random_sample_2d(m, s)
            points.append(Point(str(label)+'_'+str(i), label, x))
    
    return points


######################################################################
# k-means and k-medoids
######################################################################

def random_init(points, k) :
    """
    Randomly select k unique elements from points to be initial cluster centers.
    
    Parameters
    --------------------
        points         -- list of Points, dataset
        k              -- int, number of clusters
    
    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    ### ========== TODO : START ========== ###
    # part c: implement (hint: use np.random.choice)
    i = 1
    centers = np.random.choice(points, size=k, replace=False)
    for p in centers:
        p.label = i
        i += 1
    
    print "Selected Centers are:"
    for c in centers:
        print c.__str__()
    print "*"*20
    return centers

    
    ### ========== TODO : END ========== ###


def cheat_init(points) :
    """
    Initialize clusters by cheating!
    
    Details
    - Let k be number of unique labels in dataset.
    - Group points into k clusters based on label (i.e. class) information.
    - Return medoid of each cluster as initial centers.
    
    Parameters
    --------------------
        points         -- list of Points, dataset
    
    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    ### ========== TODO : START ========== ###
    # part f: implement
    clusters_dict = defaultdict(list)

    for p in points:
        clusters_dict[p.label+1].append(p)
    
    k_clusters = ClusterSet()

    for i in clusters_dict:
        k_clusters.add(Cluster(clusters_dict[i]))
    

    return k_clusters.medoids()
    ### ========== TODO : END ========== ###

def reassignClusters(points, k, centers):
    # reassign points based on cluster

    clusters_dict = defaultdict(list)
    dist = [0.0 for i in range(k)]
    for p in points:
        for i in range(k):
            # get distances from point p to centers
            dist[i] = p.distance(centers[i])
        # get the minimum distance
        min_distance = np.min(dist)
        # get the center index with minimum distance
        c = dist.index(min_distance)
        # assign center label to the point
        p.label = centers[c].label
        # insert the point to the cluster
        clusters_dict[c+1].append(p)

    # print "Clustering Assignments are:"
    # for i in range(1, k+1):
    #     print i
    #     for p in clusters_dict[i]:
    #         print p.__str__()
    #         print "Distance to center 1: ", p.distance(centers[0])
    #         print "Distance to center 2: ", p.distance(centers[1])
    #         print "Distance to center 3: ", p.distance(centers[2])
    # print "*"*20
    k_clusters = ClusterSet()

    for i in range(1, k+1):
        k_clusters.add(Cluster(clusters_dict[i]))

    return k_clusters, points
    

def kMeans(points, k, init='random', plot=False) :
    """
    Cluster points into k clusters using variations of k-means algorithm.
    
    Parameters
    --------------------
        points  -- list of Points, dataset
        k       -- int, number of clusters
        average -- method of ClusterSet
                   determines how to calculate average of points in cluster
                   allowable: ClusterSet.centroids, ClusterSet.medoids
        init    -- string, method of initialization
                   allowable: 
                       'cheat'  -- use cheat_init to initialize clusters
                       'random' -- use random_init to initialize clusters
        plot    -- bool, True to plot clusters with corresponding averages
                         for each iteration of algorithm
    
    Returns
    --------------------
        k_clusters -- ClusterSet, k clusters
    """
    
    ### ========== TODO : START ========== ###
    # part c: implement
    # Hints:
    #   (1) On each iteration, keep track of the new cluster assignments
    #       in a separate data structure. Then use these assignments to create
    #       a new ClusterSet object and update the centroids.
    #   (2) Repeat until the clustering no longer changes.
    #   (3) To plot, use plot_clusters(...).
    k_clusters = ClusterSet()

    # initialize cluster centers
    if init == "random":
        centers = random_init(points, k)
    if init == "cheat":
        centers = cheat_init(points)

    print "Get initial centroids:"
    for p in centers:
        print p.__str__()
    print "*" * 20

    k_clusters, points = reassignClusters(points, k, centers)

    if plot:
        plot_clusters(k_clusters, "Iteration 1", ClusterSet.centroids)

    i = 2
    while True:
        centroidsNew = k_clusters.centroids()
        
        print "New Centers are:"
        for c in centroidsNew:
            print c.__str__()
        print "*" * 20

        isConverge = True
        for j in range(k):
            if not centroidsNew[j] == centers[j]:
                isConverge = False
    
        if isConverge:
            break

        k_clusters, points = reassignClusters(points, k, centroidsNew)

        if plot:
            plot_clusters(k_clusters, "Iteration "+str(i), ClusterSet.centroids)
            i += 1

        for j in range(k):
            centers[j] = centroidsNew[j]
    
    return k_clusters
    ### ========== TODO : END ========== ###


def kMedoids(points, k, init='random', plot=False) :
    """
    Cluster points in k clusters using k-medoids clustering.
    See kMeans(...).
    """
    ### ========== TODO : START ========== ###
    # part e: implement
    k_clusters = ClusterSet()

    # initialize cluster centers
    if init == "random":
        centers = random_init(points, k)
    if init == "cheat":
        centers = cheat_init(points)

    print "Get initial centroids:"
    for p in centers:
        print p.__str__()
    print "*" * 20

    k_clusters, points = reassignClusters(points, k, centers)

    if plot:
        plot_clusters(k_clusters, "Iteration 1", ClusterSet.medoids)

    i = 2
    while True:
        centroidsNew = k_clusters.medoids()
        
        print "New Centers are:"
        for c in centroidsNew:
            print c.__str__()
        print "*" * 20

        isConverge = True
        for j in range(k):
            if not centroidsNew[j] == centers[j]:
                isConverge = False
    
        if isConverge:
            break

        k_clusters, points = reassignClusters(points, k, centroidsNew)

        if plot:
            plot_clusters(k_clusters, "Iteration "+str(i), ClusterSet.medoids)
            i += 1

        for j in range(k):
            centers[j] = centroidsNew[j]
    
    return k_clusters
    ### ========== TODO : END ========== ###


######################################################################
# main
######################################################################

def main() :
    
    ### ========== TODO : START ========== ###
    # part d, part e, part f: cluster toy dataset
    np.random.seed(1234)
    
    points = generate_points_2d(20)
    print "Initial Points:"
    for p in points:
        print p.__str__()

    print "*" * 20
    # kMeans(points, 3, init="cheat", plot=True)
    kMedoids(points, 3, init="cheat", plot=True)
    ### ========== TODO : END ========== ###
    
    


if __name__ == "__main__" :
    main()

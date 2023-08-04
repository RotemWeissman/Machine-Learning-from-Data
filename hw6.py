import numpy as np

def get_random_centroids(X, k):
    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''
    centroids = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    indices = np.random.choice(X.shape[0], size=k, replace=False)

    centroids = X[indices]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # make sure you return a numpy array
    return np.asarray(centroids).astype(np.float) 

def lp_distance(X, centroids, p=2):
    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` that holds the distances of
    all points in RGB space from all centroids
    '''
    distances = []
    k = len(centroids)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for centroid in centroids:
        centroid_distances = np.sum(np.abs(X - centroid)**p, axis=1)**(1/p)
        distances.append(centroid_distances)

    distances = np.array(distances)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return distances

def kmeans(X, k, p ,max_iter=100, centroids = None, eval = False):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    inertia = []
    if centroids is None:
        centroids = get_random_centroids(X, k)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for i in range(max_iter):
        if eval:
            inertia.append(np.sum(lp_distance(X, centroids, p).min(axis=0) ** 2))

        distances = lp_distance(X, centroids, p)
        classes = distances.argmin(axis=0)
        new_centroids = []

        for j in range(k):
            cluster_points = X[np.where(classes == j)[0]]
            j_centroid = cluster_points.mean(axis=0)
            new_centroids.append(j_centroid)

        new_centroids = np.asarray(new_centroids).astype(np.float)

        if np.array_equal(centroids, new_centroids):
            break

        centroids = new_centroids

    if eval:
        return inertia
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes

def kmeans_pp(X, k, p ,max_iter=100, eval = False):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = None
    centroids = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # choose centroids:
    indices = np.random.choice(X.shape[0], size=1, replace=False)
    centroids = X[indices]

    for i in range(1, k):
        distances = lp_distance(X, centroids, p)
        square_distances = np.min(distances**0.5, axis=0)
        weights = (square_distances / np.sum(square_distances)).flatten()
        next_centroid_idx = np.random.choice(X.shape[0], p=weights, size=1, replace=False)
        next_centroid = X[next_centroid_idx]
        centroids = np.vstack((centroids, next_centroid))

    # run k-means:
    if eval:
        return kmeans(X, k, p, max_iter, centroids, eval=True)

    centroids, classes = kmeans(X, k, p, max_iter, centroids)


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes

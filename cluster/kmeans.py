import numpy as np
from scipy.spatial.distance import cdist
from cluster.utils import get_distance


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100, clusters: np.ndarray = None):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        self.k=k
        self.tol=tol
        self.max_iter=max_iter
        self.clusters=None
        self.nsample=None
        self.ndim=None

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        # initialize k random centroids
        (self.nsample, self.ndim)=mat.shape 
        if self.nsample==0:
            raise AttributeError("The training data is empty!")
        if self.k<2:
            raise ValueError("Cannot cluster with given k value")
        ymins=np.min(mat,axis=0)
        ymaxes=np.max(mat,axis=0)
        #print(f"ymins are {ymins} and ymaxes are {ymaxes}")
        centroids=[]
        for ki in range(self.k):
            this_centroid=[]
            for yi in range(self.ndim):
                this_centroid.append(np.random.uniform(ymins[yi], ymaxes[yi]))
            centroids.append(this_centroid)
        #print(f"{len(centroids)} centroids were initialized at {centroids}")
        # iterate over all data and find the closest centroid, assign to clusters
        stop_clustering=False
        n_iter=0
        while not stop_clustering:
            assignments=np.ndarray((self.nsample,1))
            for xi in range(self.nsample):
                sample=mat[xi]
                closest_centroid_dist=np.inf
                closest_centroid=None
                for ki in range(self.k):
                    dist=get_distance(sample,centroids[ki])
                    if dist < closest_centroid_dist:
                        closest_centroid_dist=dist
                        closest_centroid=ki
                        #print(f"Updating sample {xi} to be in cluster {ki}")
                assignments[xi]=closest_centroid
            
            # pass each cluster to centroids() and get centroids
            centroids=self.get_centroids(mat,assignments)
            #(f"moving centroids to {centroids}")
            
            # pass to get_error and compare to tolerance
            SSE=self.get_error(mat,centroids,assignments)
            n_iter+=1
            # decide whether to keep iterative
            if (SSE < self.tol) or (n_iter > self.max_iter):
                stop_clustering = True
        self.centroids=centroids
        #print(f"The new centroids are {self.centroids}")
                
                
    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        (self.nsample, ndim)=mat.shape
        if self.ndim != ndim:
            raise AttributeError("Training and test data must have same number of features")
        if self.nsample==0:
            raise AttributeError("The test data is empty!")
        #print(f"predicting with centroids {self.centroids}")
        assignments=np.ndarray((self.nsample,1))
        (nk,a)=self.centroids.shape
        for xi in range(self.nsample):
            sample=mat[xi]
            closest_centroid_dist=np.inf
            closest_centroid=None
            for ki in range(nk):
                dist=get_distance(sample,self.centroids[ki])
                if dist < closest_centroid_dist:
                    closest_centroid_dist=dist
                    closest_centroid=ki
            assignments[xi]=closest_centroid
        return assignments

    def get_error(self, mat, centroids,assignments) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        SSE=0
        for xi in range(self.nsample):
            sample=mat[xi]
            sample_centroid=centroids[int(assignments[xi])]
            dist=get_distance(sample,sample_centroid)
            SSE += (dist**2)
        return SSE

    def get_centroids(self, mat, assignments) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        centroid_sums=np.zeros((self.k,self.ndim))
        centroid_counts=np.zeros((self.k,1))
        for xi in range(self.nsample): #iterate over all data and accumulate sums and counts to calculate means later
            assigned_centroid=int(assignments[xi])
            centroid_sums[assigned_centroid] += mat[xi]
            centroid_counts[assigned_centroid] += 1
        centroid=np.ndarray((self.k,self.ndim))
        
        for ki in range(self.k): #now calculate means
            centroid[ki]=centroid_sums[ki]/centroid_counts[ki]
        
        return centroid

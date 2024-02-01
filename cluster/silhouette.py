import numpy as np
from scipy.spatial.distance import cdist
from cluster.utils import get_distance


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations. For this, we need two metrics:
        - mean intra-cluster distance between a point and its cluster
        - mean inter-cluster distance between a point and its nearest other cluster

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        (nx, ny)=X.shape
        cluster_labels=np.unique(y)
        k=len(cluster_labels)
        #print(f"The {k} cluster labels are {cluster_labels}")
        scores=np.zeros((nx,1))
        for xi in range(nx):
            a_sum=0
            a_num=0
            bs_sum=np.zeros((k,1))
            bs_count=np.zeros((k,1))
            xi_cluster=y[xi]
            for xj in range(nx):
                if (y[xj]==xi_cluster) and (xi != xj):
                    a_sum+=get_distance(X[xi],X[xj])
                    a_num+=1
                elif (xi != xj):
                    for idx, other_label in enumerate(cluster_labels):
                        if y[xj]==other_label:
                            dist=get_distance(X[xi],X[xj])
                            bs_sum[idx]+=dist
                            bs_count+=1
            a=a_sum/a_num
            bs=[sums/counts for sums,counts in zip(bs_sum, bs_count)]
            bs=[x for x in bs if x!=0]
            b=min(bs)
            scores[xi]=(b-a)/max(a,b)
            #print(f"A is {a} and B is {b}")
        #print(f"The scores are {scores}")
        return scores
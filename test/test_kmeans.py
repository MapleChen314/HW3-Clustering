# Write your k-means unit tests here
import pytest
import sklearn
import numpy as np
from cluster import (
        KMeans, 
        Silhouette, 
        make_clusters,
        plot_clusters,
        plot_multipanel)

def test_check_scores():
    #Make sure 4 clusters are returned
    clusters, labels = make_clusters(k=4, scale=1,seed=15)
    km = KMeans(k=4)
    km.fit(clusters)
    pred = km.predict(clusters)
    scores = Silhouette().score(clusters, pred)
    assert len(np.unique(pred)) == 4

def test_check_scores_6():
    #Make sure 6 clusters are returned
    clusters, labels = make_clusters(k=4, scale=1,seed=17)
    km = KMeans(k=6)
    km.fit(clusters)
    pred = km.predict(clusters)
    scores = Silhouette().score(clusters, pred)
    assert len(np.unique(pred)) == 6
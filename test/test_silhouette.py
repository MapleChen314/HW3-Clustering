# write your silhouette score unit tests here
import pytest
import numpy as np
from sklearn.metrics import silhouette_score
from cluster import (
        KMeans, 
        Silhouette, 
        make_clusters,
        plot_clusters,
        plot_multipanel)

def test_check_scores():
    # Make sure all silhouette scores are within bounds
    clusters, labels = make_clusters(k=4, scale=1,seed=50)
    km = KMeans(k=4)
    km.fit(clusters)
    pred = km.predict(clusters)
    scores = Silhouette().score(clusters, pred)
    #plot_multipanel(clusters, labels, pred, scores)
    assert max(scores)<=1
    assert min(scores)>=-1

def test_sklearn():
    # Make sure my score and sklearn's silhouette score are within an order of magnitude of each other
    clusters, labels = make_clusters(k=4, scale=1,seed=12)
    km = KMeans(k=4)
    km.fit(clusters)
    pred = km.predict(clusters)
    my_scores = Silhouette().score(clusters, pred)
    my_score=float(sum(my_scores)/len(my_scores))
    sk_score=silhouette_score(clusters, np.ravel(pred))
    assert (my_score/sk_score) < 10
    assert (my_score/sk_score) > 0.1
    
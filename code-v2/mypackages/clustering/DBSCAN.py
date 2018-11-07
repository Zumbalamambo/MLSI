from sklearn.cluster import DBSCAN
import numpy as np

def getCluster(X,eps=3,
            min_samples=2, metric='euclidean', 
            etric_params=None, algorithm='auto', 
            leaf_size=30, p=None, n_jobs=None):

    clustering = DBSCAN(eps, min_samples,metric, 
            etric_params, algorithm, 
            leaf_size, p, n_jobs).fit(X)
    return clustering.labels_
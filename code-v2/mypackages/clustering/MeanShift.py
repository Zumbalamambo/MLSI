from sklearn.cluster import MeanShift
import numpy as np

#NOTE: bandwidth dictates the size of the region to search through
# if it is not set, it will automatically calculated
def getCluster(X,bandwidth=None,
            seeds=None, bin_seeding=False, 
            min_bin_freq=1, cluster_all=True, n_jobs=None):
    clustering = MeanShift(bandwidth, bin_seeding, 
            min_bin_freq, cluster_all, n_jobs).fit(X)
    return clustering.labels_


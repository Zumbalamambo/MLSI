from sklearn.cluster import AffinityPropagation
import numpy as np

#TODO: AffinityAlgorithm can provide the clustering centers
def getCluster(org_data,amping=0.5, max_iter=200, convergence_iter=15,\
    copy=True, preference=None, affinity='euclidean',verbose=False):
    
    clustering = AffinityPropagation(amping, max_iter, convergence_iter,\
        copy, preference, affinity,verbose).fit(org_data)
    clustering

    return clustering.labels_
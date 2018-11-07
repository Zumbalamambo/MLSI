from sklearn.cluster import AgglomerativeClustering
import numpy as np


#TODO: try “ward”, “complete”, “average”
def getCluster(X,n_clusters=2,
            affinity='euclidean', memory=None, 
            connectivity=None, compute_full_tree='auto', 
            linkage='ward', pooling_func='deprecated'):
    
    clustering = AgglomerativeClustering(n_clusters,
            affinity, memory, 
            connectivity, compute_full_tree, 
            linkage, pooling_func).fit(X)
    clustering 
    return clustering.labels_
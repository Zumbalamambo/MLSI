from sklearn.cluster import SpectralClustering
import numpy as np

#TODO: if need finer details, switch the strategy to "kmeans"
def getCluster(org_data,n_clusters=4,
            assign_labels="discretize",
            random_state=0):
    X = org_data
    clustering = SpectralClustering(n_clusters,
            assign_labels,
            random_state).fit(X)
    return clustering.labels_
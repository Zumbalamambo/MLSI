from sklearn.cluster import KMeans

def getCluster(data,n_clusters,n_init):
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=n_init)
    kmeans.fit(data)
    #NOTE: labels start from 0
    labels=kmeans.labels_
    return labels
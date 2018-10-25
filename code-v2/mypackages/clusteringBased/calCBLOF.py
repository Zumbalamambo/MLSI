import numpy as np 

#calculate boundary cluster and label the type
def getBoundary(total_size,clusters,alpha,beta):
    temp_size_sum=0
    boundary=None
    for i in range(len(clusters)):
        if boundary != None:
            break
        if i==len(clusters)-1:
            boundary=i
            break
        temp_size_sum+=clusters[i].size
        if temp_size_sum >= total_size*alpha \
            and clusters[i].size/clusters[i+1].size>=beta:
            boundary=i
            break
    for i in range(len(clusters)):
        if i<=boundary:
            clusters[i].change_type("large")
        else:
            clusters[i].change_type("small")
    print("the boundary of large clusters is",boundary)
    return boundary,clusters

#calculate the centorid of a cluster
def getCentroid(cluster_data):
    centroid=np.mean(cluster_data,axis=0)#calculate the mean of each column(feature)
    return centroid

#calculate the L2 distance between two points
def calDistance(p1,p2):
    return np.linalg.norm(p1-p2,axis=-1)

def FindCBLOF(data,clusters,alpha,beta,isUnweighted=False):
    #sort the cluster by size, TODO:maybe label is unneccessary
    clusters_sorted=sorted(clusters, key = lambda x: x.size, reverse=True)
    boundary,clusters_labeled=getBoundary(total_size=data.shape[0],clusters=clusters_sorted,alpha=alpha,beta=beta)
    #clean memory
    clusters=None
    clusters_sorted=None
    #calculate centroids of large clusters
    centroids=np.zeros(shape=(boundary+1,data.shape[1]))
    for i in range(boundary+1):
        centroids[i]=getCentroid(data[clusters_labeled[i].tuples])
    
    data_factor=np.zeros(data.shape[0])
    if isUnweighted:#neglect the weight
        for i in range(len(clusters_labeled)):
            print("calculating uCBLOF in cluster %d ... the size is %d" %(i,clusters_labeled[i].size))
            #for data in large clusters
            if clusters_labeled[i].cluster_type=="large":#index boundary itself is not large cluster
                for j in range(clusters_labeled[i].size):
                    temp_member=clusters_labeled[i].tuples[j]
                    data_factor[temp_member]=calDistance(data[temp_member],centroids[i])
            #for data in small clusters   
            else:
                for j in range(clusters_labeled[i].size):
                    temp_member=clusters_labeled[i].tuples[j]
                    distances=np.linalg.norm(data[temp_member]-centroids,axis=-1)
                    #get the closest one, np.argmin return the index of 1-dim array
                    closest_cluster=np.argmin(distances)
                    data_factor[temp_member]=calDistance(data[temp_member],centroids[closest_cluster])
    else:#calculate CBLOF
        for i in range(len(clusters_labeled)):
            print("calculating CBLOF in cluster %d ... the size is %d" %(i,clusters_labeled[i].size))
            #CBLOF for data in large clusters
            if clusters_labeled[i].cluster_type=="large":#index boundary itself is not large cluster
                for j in range(clusters_labeled[i].size):
                    temp_member=clusters_labeled[i].tuples[j]
                    data_factor[temp_member]=clusters_labeled[i].size*\
                        calDistance(data[temp_member],centroids[i])
            #CBLOF for data in small clusters   
            else:
                for j in range(clusters_labeled[i].size):
                    temp_member=clusters_labeled[i].tuples[j]
                    distances=np.linalg.norm(data[temp_member]-centroids,axis=-1)
                    #np.argmin return the index of 1-dim array
                    closest_cluster=np.argmin(distances)
                    data_factor[temp_member]=clusters_labeled[i].size*\
                        calDistance(data[temp_member],centroids[closest_cluster])
    return data_factor
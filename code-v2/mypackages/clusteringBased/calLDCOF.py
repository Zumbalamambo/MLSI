import numpy as np
from . import clusterStructure as cs
from . import calCBLOF

def getClusterAvgDis(data,clusters,centroids):
    clusters_AvgDis=[None]*len(clusters)
    for i in range(len(clusters)):
        tem_centroid=centroids[i]
        tem_dis=0
        for j in clusters[i].tuples:
            tem_dis+=calCBLOF.calDistance(tem_centroid,data[j])
        clusters_AvgDis[i]=tem_dis/clusters[i].size
    return clusters_AvgDis

def findLDCOF(data,labels,n_clusters,alpha,beta):
    
    clusters=list()
    for i in range(n_clusters):
        clusters.append(cs.ClusterStructure(data.shape[1]))
    for i in range(data.shape[0]):
        clusters[labels[i]].mem_append(i,data[i],calSummary=False)
    #sort by size
    clusters_sorted=sorted(clusters, key = lambda x: x.size, reverse=True)    
    #to tell who is large
    boundary,clusters_classified=calCBLOF.getBoundary(data.shape[0],clusters_sorted,alpha,beta)
    #calculate the centroids for later use
    centroids=np.zeros(shape=(len(clusters),data.shape[1]))
    for i in range(len(clusters)):
        centroids[i]=calCBLOF.getCentroid(data[clusters_classified[i].tuples])
        print("the size of the",i,"cluster is:",clusters_classified[i].size)
        
    centroids_large=centroids[0:boundary+1]
    #mediate step
    clusters_avg_dis=getClusterAvgDis(data,clusters_classified,centroids)
    #clear memory
    clusters=None
    clusters_sorted=None
    # last step
    data_LDCOF=np.ndarray(shape=(data.shape[0],1))
    data_count=0
    for i in range(len(clusters_classified)):
        for j in range(clusters_classified[i].size):
            data_count+=1
            if data_count%50000 ==0:
                print("calculating the LDCOF of %d training data..." %(data_count))
            
            temp_member=clusters_classified[i].tuples[j]
            if i<=boundary:
                tem_LDCOF=calCBLOF.calDistance(data[temp_member],centroids[i])/clusters_avg_dis[i]
            else:
                distances=np.linalg.norm(data[temp_member]-\
                    centroids_large,axis=-1)
                #get the closest one, np.argmin return the index of 1-dim array
                closest_cluster=np.argmin(distances)
                tem_LDCOF=distances[closest_cluster]/clusters_avg_dis[closest_cluster]
            data_LDCOF[temp_member]=tem_LDCOF
    return data_LDCOF
from mypackages.processing import DataProcess
from mypackages.processing import GeoProcess

from mypackages.processing import array_trs as art
from mypackages.processing import open_image as oi

from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

import numpy as np
import time


def extract_compareClustering(clusterClass):
    # Get data, n_bands=4
    norm_img_path="C:\\Users\\DELL\\Projects\\MLS_cluster\\image-v2-timeseries\\newest"
    img="4Band_Subtracted_20040514_20050427"
    
    dataset=oi.open_tiff(norm_img_path,img)
    H=dataset[1]
    W=dataset[2]
    n_bands=dataset[3]
    org_data=art.tif2vec(dataset[0])#NOTE: this step is really important

    select_path="C:\\Users\\DELL\\Projects\\MLS_cluster\\image-v2-timeseries\\EXTRACT"
    select_img="SOMOCLU_20_20_HDBSCAN_cl_2_2004_2005_min_cluster_size_4_alg_best_"
    simg=oi.open_tiff(select_path,select_img)
    select=simg[0]#(2720000)

    changePos=DataProcess.selectArea(select,n_bands,-1,isStack=True)
    ns_changePos=DataProcess.selectArea(select,n_bands,-1,isStack=False)
    ns_nonChangePos=DataProcess.selectArea(select,n_bands,0,isStack=False)
    
    X_train=org_data[changePos].reshape(-1,n_bands)
    
    result=np.zeros_like(select.reshape(-1,1))
    
    for cls_name,cls_class in clusterClass.items():
        print("running",cls_name,"...")
        t0=time.clock()
        cls_class.fit(X_train)
        usingTime=time.clock()-t0

        # combine the result
        result[ns_changePos]=cls_class.labels_
        result[ns_nonChangePos]=np.max(cls_class.labels_)+1

        evaluation=silhouette_score(X=org_data,
            labels=result,metric='euclidean',sample_size=10000)

        save_path="C:\\Users\\DELL\\Projects\\MLS_cluster\\image-v2-timeseries\\sklearn_clustering\\compare"
        DataProcess.visualize_class(result.reshape(H,W),save_path+'\\'+cls_name+"_change_area_class")
        
        # save using time
        print("save the information to txt file...")
        with open(save_path+'/'+"Outlier Detection Algorithms Running Time.txt", 'a') as f:
            f.write("detetion algorithm: "+cls_name+
                    "\nsilhouette_score:"+str(evaluation)+
                    "\ndetection using time: "+ str(usingTime))
            f.write("\n----------------------------------------------\n")
    
if __name__ == "__main__":

    n_clusters=3
    clusterClass={
        # "Affinity": #NOTE: memory error confirmed
        #     AffinityPropagation(damping=.5,
        #         max_iter=200, convergence_iter=15, 
        #         copy=True, preference=None, 
        #         affinity='euclidean', verbose=False),
        # "Agglomerative": #NOTE: memory error confirmed
        #     AgglomerativeClustering(n_clusters=n_clusters,
        #         affinity='euclidean', memory=None, 
        #         connectivity=None, compute_full_tree='auto', 
        #         linkage='ward', pooling_func='deprecated'),
        "Birch":
            Birch(threshold=0.5, branching_factor=50,  
                n_clusters=n_clusters, compute_labels=True,copy=True),
        # "DBSCAN":#NOTE:looks like memory error occur on this
        #     DBSCAN(eps=0.5, min_samples=5, 
        #         metric='euclidean', metric_params=None,
        #         algorithm='auto', leaf_size=30, p=None, n_jobs=1),
        "kMeans":
            KMeans(n_clusters=n_clusters, init='k-means++', 
                n_init=10, max_iter=300, tol=1e-4, 
                precompute_distances='auto', 
                verbose=0, random_state=None, 
                copy_x=True, n_jobs=1, algorithm='auto'),
        # "MeanShift":#NOTE: Slow confirmed
        #     MeanShift(bandwidth=None, seeds=None, 
        #         bin_seeding=False, min_bin_freq=1, 
        #         cluster_all=True, n_jobs=1),
        # "Spectral":#NOTE: memory error confirmed
        #     SpectralClustering(n_clusters=n_clusters, eigen_solver=None, 
        #         random_state=None, n_init=10, 
        #         gamma=1., affinity='rbf', 
        #         n_neighbors=10, eigen_tol=0.0, 
        #         assign_labels='kmeans', degree=3, 
        #         coef0=1, kernel_params=None, n_jobs=1),
    }
    
    extract_compareClustering(clusterClass)
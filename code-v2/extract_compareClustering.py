from mypackages.processing import DataProcess
from mypackages.processing import GeoProcess
from mypackages import clustering as cl
from mypackages import clusteringBased as cb
from mypackages import scoresToResults as sc2r
from mypackages.processing import open_image as oi

import numpy as np
import time


def selectRun():
    norm_img_path="C:\\Users\\DELL\\Projects\\MLS_cluster\\image-v2-timeseries\\Encoded_dataset\\Encoded_models_2018-10-03_1339\\subtracted_norm_from_norm"
    norm_data_path="C:\\Users\\DELL\\Projects\\MLS_cluster\\image-v2-timeseries\\raw_data\\1339_sub"
    img="Subtracted_20040514_20050427"
    raw="Subtracted_20040514_20050427_raw_data"
    org_data=DataProcess.csv_to_array(norm_data_path,raw)
    n_bands=3
    
    select_path="C:\\Users\\DELL\\Projects\\MLS_cluster\\image-v2-timeseries\\EXTRACT"
    select_img="SOMOCLU_20_20_HDBSCAN_cl_2_2004_2005_min_cluster_size_4_alg_best_"
    simg=oi.open_tiff(select_path,select_img)
    select=simg[0]
    
    changePos=DataProcess.selectArea(select,n_bands,-1,isStack=True)
    ns_changePos=DataProcess.selectArea(select,n_bands,-1,isStack=False)
    ns_nonChangePos=DataProcess.selectArea(select,n_bands,0,isStack=False)

    result=np.zeros_like(select.reshape(-1,1))

    changeArea=org_data[changePos].reshape(-1,n_bands)
    # change_label=cl.kMeans.getCluster(changeArea,3,5) # n_cluster=3
    # change_label=cl.MeanShift.getCluster(changeArea,None,None,False,1,True,None)
    # change_label=cl.BIRCH.getCluster(changeArea,threshold=0.6, branching_factor=50, n_clusters=4, compute_labels=True, copy=True)
    change_label=cl.DBSCAN.getCluster(changeArea)#default result
    result[ns_changePos]=change_label
    result[ns_nonChangePos]=np.max(change_label)+1


    DataProcess.int_to_csv(select_path,select_img,result,"_meanshift_mark_change_area_class")
    DataProcess.visualize_class(select_path,select_img,select_path,select_img+"_meanshift_mark_change_area_class")

    #FIXME:---------------------------------------
    # Traceback (most recent call last):
    #   File "codev3.py", line 50, in <module>
    #     selectRun()
    #   File "codev3.py", line 44, in selectRun
    #     d_score=cb.calLDCOF.findLDCOF(org_data,result,4,0.7,2)
    #   File "C:\Users\DELL\Projects\MLS_cluster\code_g\code-v2\mypackages\clusteringBased\calLDCOF.py", line 21, in findLDCOF
    #     clusters[labels[i]].mem_append(i,data[i],calSummary=False)
    #   TypeError: only integer scalar arrays can be converted to a scalar index
    #---------------------------------------------
    # d_score=cb.calLDCOF.findLDCOF(org_data,result,4,0.7,2)
    # outlier_label=sc2r.highRank.getOutliers(d_score,98)
    
    # DataProcess.int_to_csv(select_path,select_img,outlier_label,"_mark_change_area_outlier")
    # GeoProcess.getSHP(select_path,select_img,select_path,"extracted_kmeans",result)

selectRun()

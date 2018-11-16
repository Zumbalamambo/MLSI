# -*- coding: utf-8 -*-
"""Example of using HBOS for outlier detection
"""
# Author: Yue Zhao <yuezhao@cs.toronto.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys
import time

from sklearn.utils import check_X_y
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pyod.models.abod import ABOD
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA

from mypackages.processing import DataProcess
from mypackages.processing import GeoProcess
from mypackages.processing import open_image as oi
from mypackages.processing import array_trs as art

import numpy as np

def RunPyodOutlier(classifiers,outlier_save_path,isExtract=True):
    # Get data, n_bands=4
    norm_img_path="C:\\Users\\DELL\\Projects\\MLS_cluster\\image-v2-timeseries\\newest"
    img="4Band_Subtracted_20040514_20050427"
    
    dataset=oi.open_tiff(norm_img_path,img)
    H=dataset[1]
    W=dataset[2]
    n_bands=dataset[3]
    org_data=art.tif2vec(dataset[0])#NOTE: this step is really important

    #TODO: normalize the data?
    
    if isExtract:
        # extract out the changed area
        select_path="C:\\Users\\DELL\\Projects\\MLS_cluster\\image-v2-timeseries\\EXTRACT"
        select_img="SOMOCLU_20_20_HDBSCAN_cl_2_2004_2005_min_cluster_size_4_alg_best_"
        simg=oi.open_tiff(select_path,select_img)
        select=simg[0]#(2720000)

        changePos=DataProcess.selectArea(select,n_bands,-1,isStack=True)
        ns_changePos=DataProcess.selectArea(select,n_bands,-1,isStack=False)
        ns_nonChangePos=DataProcess.selectArea(select,n_bands,0,isStack=False)

        X_train=org_data[changePos].reshape(-1,n_bands)
        print("shape of original data: ",org_data.shape)
        print("shape of extracted data: ",X_train.shape)
        # to save the final result
        outlier_result=np.zeros_like(select.reshape(-1,1))
        score_result=np.empty_like(select.reshape(-1,1))
    else:
        X_train=org_data.reshape(-1,n_bands)
        print("shape of training data: ",X_train.shape)

    
    for clf_name,clf in classifiers.items():
        if not isExtract:
            clf_name="no_extract_"+clf_name
        
        print("running "+clf_name+"...")
        t0=time.clock()
        clf.fit(X_train)
        usingTime=time.clock()-t0
        # get the prediction labels and outlier scores of the training data
        y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
        y_train_scores = clf.decision_scores_  # raw outlier scores
        
        if isExtract:
            # combine the extraction non-changed label&&scores and the algorithm result
            outlier_result[ns_changePos]=y_train_pred
            outlier_result[ns_nonChangePos]=0
            score_result[ns_changePos]=DataProcess.scaleNormalize(y_train_scores,(0,500)).reshape(-1,)
            score_result[ns_nonChangePos]=0
                        #save the outlier detection result as .tif and .shp file
        else:
            # combine the extraction non-changed label and the algorithm result
            outlier_result=y_train_pred
            score_result=DataProcess.scaleNormalize(y_train_scores,(0,500)).reshape(-1,)
        
        print("the scale of the y_train_score is:",y_train_scores.min(),y_train_scores.max())
        print("the scale of the score_result is:",score_result.min(),score_result.max())

        DataProcess.int_to_csv(outlier_save_path,img,outlier_result,clf_name+"_outliers")
        GeoProcess.getSHP(norm_img_path,img,outlier_save_path,clf_name+"_outliers",outlier_result)
        
        #save the outlier scores as heatmap
        DataProcess.saveHeatMap(score_result.reshape(H,W),outlier_save_path+"\\"+clf_name)

        print("save the information to txt file...")
        with open(outlier_save_path+'/'+"Outlier Detection Algorithms Running Time.txt", 'a') as f:
            f.write("detetion algorithm: "+clf_name+
                    "\ndetection using time: "+ str(usingTime))
            f.write("\n----------------------------------------------\n")

if __name__ == "__main__":
    outlier_save_path="C:\\Users\\DELL\\Projects\\MLS_cluster\\image-v2-timeseries\\Pyod_Algorithms\\new_compare_extraction"

    outliers_fraction=0.005
    random_state=np.random.RandomState(2018)
    classifiers = {
                'Cluster-based Local Outlier Factor (CBLOF)':
                    CBLOF(contamination=outliers_fraction,
                            check_estimator=False, random_state=random_state),#
                'Histogram-base Outlier Detection (HBOS)': HBOS(
                    contamination=outliers_fraction),
                'Isolation Forest': IForest(contamination=outliers_fraction,
                                            random_state=random_state),
                'K Nearest Neighbors (KNN)': KNN(
                    contamination=outliers_fraction),
                'Average KNN': KNN(method='mean',
                                    contamination=outliers_fraction),
                'Median KNN': KNN(method='median',
                                    contamination=outliers_fraction),
                'Local Outlier Factor (LOF)':
                    LOF(n_neighbors=35, contamination=outliers_fraction),
                'Minimum Covariance Determinant (MCD)': MCD(
                    contamination=outliers_fraction, random_state=random_state),
                # 'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction, #NOTE: slow, never try again
                #                                 random_state=random_state),
                'Principal Component Analysis (PCA)': PCA(
                    contamination=outliers_fraction, random_state=random_state),
                'AutoEncoder':
                  AutoEncoder(epochs=2, hidden_neurons=[4,2,4],contamination=outliers_fraction),
                'Feature Bagging':
                    FeatureBagging(LOF(n_neighbors=35),
                                    contamination=outliers_fraction,
                                    check_estimator=False,
                                    random_state=random_state),
                'Angle-based Outlier Detector (ABOD)':
                    ABOD(n_neighbors=10,
                            contamination=outliers_fraction),
            }
    
    RunPyodOutlier(classifiers,outlier_save_path,isExtract=True)
    # RunPyodOutlier(classifiers,outlier_save_path,isExtract=False)

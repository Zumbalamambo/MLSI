from . import open_image as oi
from . import array_trs as art
from . import create as cr
from osgeo import gdal,gdal_array

from . import DataProcess
import numpy as np

def getTIF(img_path,img_name,save_path,extend_name,result_array):
    img=oi.open_tiff(img_path,img_name)
    H,W=img[1],img[2]
    dats = cr.create_tiff(nb_channels=4,new_tiff_name=extend_name+img_name+".tif",width = W, \
            height= H,data_array=result_array,datatype=gdal.GDT_Float64, \
            geotransformation=img[4],projection=img[5])

    #use the ds_tiff to get an outlier image file for QIGS
    #use .shp file to point out the outliers
    cr.vectorize_tiff(main_path=save_path,\
            shp_name="outlier_image_"+extend_name+img_name,ds_tiff=dats)
    dats=None

def getSHP(img_path,img_name,save_path,extend_name,result_array):
    img=oi.open_tiff(img_path,img_name)
    H,W=img[1],img[2]
    result_array=result_array.reshape(H,W)
    dats = cr.create_tiff(nb_channels=1,new_tiff_name="outlier_image_"+extend_name+img_name+".tif",width = W, \
            height= H,data_array=result_array,datatype=gdal.GDT_UInt16, \
            geotransformation=img[4],projection=img[5])

    #use the ds_tiff to get an outlier image file for QIGS
    #use .shp file to point out the outliers
    cr.vectorize_tiff(main_path=save_path,\
            shp_name="outlier_image_"+extend_name+img_name,ds_tiff=dats)
    dats=None

def getExtractionData():
    # Get data, n_bands=4
    norm_img_path="C:\\Users\\DELL\\Projects\\MLS_cluster\\image-v2-timeseries\\newest"
    img="4Band_Subtracted_20040514_20050427"

    dataset=oi.open_tiff(norm_img_path,img)
    H=dataset[1]
    W=dataset[2]
    n_bands=dataset[3]
    org_data=art.tif2vec(dataset[0])#NOTE: this step is really important

    #NOTE: Normalize the scale of the orignialdata
    org_data = org_data / org_data.max(axis=0)

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

    return X_train,outlier_result,score_result,changePos,ns_changePos,ns_nonChangePos,n_bands,H,W
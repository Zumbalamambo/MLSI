import open_image as oi 
import create as cr 
import array_trs as art

from osgeo import gdal,gdal_array

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
# import matplotlib.pyplot as plt

img=oi.open_tiff("C:\\Users\\DELL\\Projects\\VHS","test")
# print(img[0].shape)
H,W=img[1],img[2]
data=art.tif2vec(img[0])
# #cluster
kmeans=KMeans(init='k-means++', n_clusters=5,n_init=3)
kmeans.fit(data)
#transform 1D array into 2D matrix
result_array=kmeans.labels_.reshape(H,W)
#use a label matrix to create a ds_diff
dats = cr.create_tiff(nb_channels=1,new_tiff_name="outputCluster.tif",width = W, \
        height= H,data_array=result_array,datatype=gdal.GDT_UInt16, \
        geotransformation=img[4],projection=img[5])
#use the ds_tiff to get an outlier image file for QIGS
#TODO: still, don't know where to use .shp file
cr.vectorize_tiff(main_path="C:\\Users\\DELL\\Projects\\VHS",\
        shp_name="cluster_outlier",ds_tiff=dats)
dats=None



import open_image as oi 
import create as cr 
import array_trs as art

from osgeo import gdal,gdal_array

import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

import pprint as pp

img=oi.open_tiff("C:\\Users\\DELL\\Projects\\MLS_cluster","Subtracted_2002_2004_sample")
print(img[0].shape)
H,W=img[1],img[2]
data=art.tif2vec(img[0])
data_l1=sklearn.preprocessing.normalize(data,norm='l1')
data_l2=sklearn.preprocessing.normalize(data,norm='l2')
data_l1=art.percision_trans(data_l1,"digit_int",100)
data_l2=art.percision_trans(data_l2,"digit_int",100)
pp.pprint(data_l1[:5000])
pp.pprint(data_l2[:5000])
# #cluster
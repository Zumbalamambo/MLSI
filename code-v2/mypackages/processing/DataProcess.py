from osgeo import gdal,gdal_array
import numpy as np
from numpy import genfromtxt
import os

from . import create as cr
from . import open_image as oi 
from . import array_trs as art

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import pandas as pd 
from pandas.plotting import scatter_matrix


#dirPath="C:\\Users\\DELL\\Projects\\MLS_cluster"
#fileName="Encoded_20040514"
def img_subtract(dirPath,fileName):
    img_2004=oi.open_tiff(dirPath,fileName)
    img_2002=oi.open_tiff(dirPath,fileName)

    sub_array=np.subtract(img_2004[0],img_2002[0],dtype="float32")

    dats = cr.create_tiff(nb_channels=img_2004[3],new_tiff_name="subtracted_v1.tif",\
        width = img_2004[2],height= img_2004[1],data_array=sub_array,datatype=gdal.GDT_Float32, \
        geotransformation=img_2004[4],projection=img_2004[5])

    dats=None

def img_to_csv(dirpath,savepath,file_name):
    img=oi.open_tiff(dirpath,file_name)# the former one end without "\\"
    data=art.tif2vec(img[0])
    array_to_csv(savepath,file_name,data,"_raw_data")

def csv_to_array(file_path,load_name):
    a=np.loadtxt(file_path+"/"+load_name+".csv",delimiter=';')#TODO: notice the delimiter
    # a=a.astype(np.float)
    # genfromtxt(file_path+"/"+load_name+".csv", delimiter=';')
    return a

def array_to_csv(save_path,save_name,data,extend_name=''):
    np.savetxt( save_path+"/"+save_name+extend_name+".csv",data,fmt="%.8f",delimiter=';')

def int_to_csv(save_path,save_name,data,extend_name=None):
    np.savetxt( save_path+"/"+save_name+extend_name+".csv",data,fmt="%d",delimiter=';')

def file_name(file_dir,extendtion):   
    L=[]
    names_no_etd=[]
    for dirpath, dirnames, filenames in os.walk(file_dir):  
        for file in filenames :  
            if os.path.splitext(file)[1] == extendtion:
                L.append(os.path.join(dirpath, file))
                names_no_etd.append(os.path.splitext(file)[0])
    return L,names_no_etd

# NOTE: transform a numpy array
def scaleNormalize(npdata,r=(0,1)):
    scaler = MinMaxScaler(feature_range=r)
    X_minmax = scaler.fit_transform(npdata.reshape(-1,1))
    return X_minmax

def visualize_class(img_path,img_name,labels_path,labels_name):
    
    img=oi.open_tiff(img_path,img_name)
    H,W=img[1],img[2]

    data_class=np.loadtxt(labels_path+"/"+labels_name+".csv")#,delimiter=';'
    data_class=data_class.astype(np.int)
    data_class=data_class.reshape((H,W))
    # print(H,W,data_class.shape)
    
    fig = plt.figure()
    plt.imshow(data_class)
    # plt.show()
    fig.savefig(labels_name+'_class_visualization.png')

# visualize multi-dimension numpy array
def showScatterPlot(npdata):
    print(npdata.shape)
    df= pd.DataFrame({'Band 1':npdata[:,0],'Band 2':npdata[:,1],
            'Band 3':npdata[:,2]})
    # sns.set(style="ticks")
    # sns.pairplot(df, hue="Bands")
    scatter_matrix(df, alpha = 0.2, diagonal = 'kde')
    plt.show() 

#save heat map of a 2-D numpy array
def saveHeatMap(npdata,save_name):
    # npdata=scaleNormalize(npdata,(0,1000))
    fig = plt.figure()
    plt.imshow(npdata,cmap=cm.hot)
    plt.colorbar()
    fig.savefig(save_name+'.png',dpi=300)

#used to select the masked area
def selectArea(selectMask,n_band,value,isStack=True):
    a=selectMask.reshape(-1,1)
    s=a
    if isStack:
        #manual stack and reshape...
        for i in range(n_band-1):
            s=np.hstack((s,a))
    y=np.where(s==value)
    return y
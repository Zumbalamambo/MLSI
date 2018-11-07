from osgeo import gdal,gdal_array
import numpy as np
from numpy import genfromtxt
import os
import matplotlib.pyplot as plt

from . import create as cr
from . import open_image as oi 
from . import array_trs as art

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
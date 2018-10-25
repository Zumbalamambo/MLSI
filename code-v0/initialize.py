#-----------pakage-------------------
#gdal
from osgeo import gdal,gdal_array
#machine learning
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
# system commander parameters read
import sys
import os 
# -----------self written--------------------
#image processing
import open_image as oi 
import create as cr 
import array_trs as art
#clustering
import squeezer
import findCBLOF as fc
import findLDCOF as fl

def get_file_name(file_dir):   
    L=[]
    f_names=[]
    for dirpath, dirnames, filenames in os.walk(file_dir):  
        for file in filenames :  
            if os.path.splitext(file)[1] == '.txt':  
                L.append(os.path.join(dirpath, file)) 
                f_names.append(os.path.splitext(file)[0]) 
    return L,f_names

def main():
        #---------information about the system args-------------------
        print("the arguments are strings and the index starts from 0")
        print("the number of system args:",len(sys.argv))
        print("the arguments are:", str(sys.argv))
        print("the type of the arguments group:",type(sys.argv))
        #------------------------------------------------------------- 
        if "try-sample" in sys.argv:
            img=oi.open_tiff("C:\\Users\\DELL\\Projects\\MLS_cluster\\QGIS-img\\","Subtracted_v1"+"_sample")
        else:
            img=oi.open_tiff("C:\\Users\\DELL\\Projects\\MLS_clusterQGIS-img\\","Subtracted_v1")
        H,W=img[1],img[2]
        data=art.tif2vec(img[0])
        print("the shape of the data is",data.shape)
        print("the size of the data is",data.size)
        #-------------------------------------------------------------  
        if "uCBLOF" in sys.argv:# MEANS sys.argv[1]=="uCBLOF"
            norm_dit=int(sys.argv[2])
            threshold=float(sys.argv[3])
            alpha=float(sys.argv[4])
            beta=float(sys.argv[5])
            percent_boundary=int(sys.argv[6])
            isUnweighted=bool(sys.argv[7])
            #---------------------------------------------------------
            # data pre processing
            data_nomal = art.data_normalize(data)
            data_nomal = art.percision_trans(data_nomal,"digit_int",norm_dit)
            np.savetxt("data_nomal.txt",data_nomal,fmt="%d")
            print("store the data after normalization...")
            #-------------------------------------------------------------
            #squeezer clustering
            #TODO:calculate the average similarity to get the parameter
            #TODO:get improve the algorithm to d-squeezer for large dataset
            squeezer_data_labels_,clusters_=squeezer.squeezer(data_array=data_nomal,nb_features=img[3],threshold=threshold)
            np.savetxt("squeezer_data_labels_.txt   ",squeezer_data_labels_,fmt="%d")
            #CBLOF
            data_CBLOF=fc.FindCBLOF(data=data_nomal,clusters=clusters_,alpha=alpha,beta=beta,isUnweighted=isUnweighted)
            np.savetxt("data_CBLOF.txt",data_CBLOF,fmt="%d")
            print("store the CBLOF of the data...")
            #np.percentile with x, means x% number will be less than the returned value
            # print(data_CBLOF.min(),data_CBLOF.max(),data_CBLOF.mean(),data_CBLOF.std(),np.percentile(data_CBLOF,5))
            b=np.percentile(data_CBLOF,percent_boundary)
            result_array=np.zeros(shape=H*W)
            result_array[np.where(data_CBLOF>b)]=1
            print("the value of percent bound is",b)
            #b = np.loadtxt('test1.txt', dtype=int)
        #------------------------------------------------------------- 
        elif "LDCOF" in sys.argv: # MEANS sys.argv[1]=="LDCOF"
            print("running LDCOF")
            n_clusters=int(sys.argv[2])
            alpha=float(sys.argv[3])
            beta=float(sys.argv[4])
            n_init=int(sys.argv[5])
            percent_boundary=int(sys.argv[6])
            data_LDCOF=fl.findLDCOF(data=data,n_clusters=n_clusters,alpha=alpha,beta=beta,n_init=n_init)
            b=np.percentile(data_LDCOF,percent_boundary)
            print("the value of percent bound is",b)
            result_array=np.zeros(shape=H*W)
            result_array[np.where(data_LDCOF>b)[0]]=1
        elif "score" in sys.argv:#argv[1]#
            percent_boundary=float(sys.argv[2])
            scores_file,f_names=get_file_name("C:\\Users\\DELL\\Projects\\MLS_cluster\\scores")
            for i in range(len(scores_file)):
                file_name=scores_file[i] 
                label_name=f_names[i] 
                score_data=np.loadtxt(file_name,dtype=float)
                b=np.percentile(score_data,percent_boundary)
                print("In",file_name,"the value of percent bound is",b)
                result_array=np.zeros(shape=H*W)
                result_array[np.where(score_data>b)[0]]=1
                result_array=result_array.reshape(H,W)
                dats = cr.create_tiff(nb_channels=1,new_tiff_name="cluster_image_"+label_name+".tif",width = W, \
                    height= H,data_array=result_array,datatype=gdal.GDT_UInt16, \
                    geotransformation=img[4],projection=img[5])

                #use the ds_tiff to get an outlier image file for QIGS
                #use .shp file to point out the outliers
                cr.vectorize_tiff(main_path="C:\\Users\\DELL\\Projects\\MLS_cluster",\
                        shp_name="cluster_outlier_"+label_name,ds_tiff=dats)
                dats=None
        else:
            print("no specific cluster type, program stop")
            exit()
        #------------create image file--------------------------------
        #use a label matrix to create a ds_diffs
        #note that the number of channel is 1 because we only need to know where is the outliers
        
        result_array=result_array.reshape(H,W)
        dats = cr.create_tiff(nb_channels=1,new_tiff_name="cluster_image_"+sys.argv[1]+".tif",width = W, \
                height= H,data_array=result_array,datatype=gdal.GDT_UInt16, \
                geotransformation=img[4],projection=img[5])

        #use the ds_tiff to get an outlier image file for QIGS
        #use .shp file to point out the outliers
        cr.vectorize_tiff(main_path="C:\\Users\\DELL\\Projects\\MLS_cluster",\
                shp_name="cluster_outlier_"+sys.argv[1],ds_tiff=dats)
        dats=None

if __name__ == "__main__":
    main()
# from processing import DataProcess as DataProcess
# import clustering
from mypackages.processing import DataProcess
from mypackages.processing import GeoProcess
from mypackages import clustering as cl
from mypackages import clusteringBased as cb
from mypackages import scoresToResults as sc2r

def runClusteringBased(img_path,img_name,data_path,data_name,outlier_save_path,\
    clusteringPara,outlierPara,o_filter="highRank"):
    #clusteringPara[0] is the name, the rest are parameters
    #TODO:change score and filter para
    org_data=DataProcess.csv_to_array(data_path,data_name)
    if clusteringPara[0]=="kMeans":
        d_label=cl.kMeans.getCluster(org_data,clusteringPara[1],clusteringPara[2])
    if clusteringPara[0]=="BIRCH":
        d_label=cl.BIRCH.getCluster(org_data,clusteringPara[1],clusteringPara[2],clusteringPara[3],clusteringPara[4])

    if outlierPara[0]=="LDCOF":
        d_score=cb.calLDCOF.findLDCOF(org_data,d_label,outlierPara[1],outlierPara[2],outlierPara[3])
    
    if o_filter=="highRank":
        outlier_label=sc2r.highRank.getOutliers(d_score,98)

    DataProcess.int_to_csv(outlier_save_path,img_name,outlier_label,"outlier_label")
    GeoProcess.getSHP(img_path,img_name,outlier_save_path,outlier_label)#FIXME: the .tif file could not be specified the path

norm_subtracted_path="C:\\Users\\DELL\\Projects\\MLS_cluster\\image-v2-timeseries\\Encoded_dataset\\Encoded_models_2018-10-03_1339\\subtracted_norm_from_norm"
norm_subtracted_save="C:\\Users\\DELL\\Projects\\MLS_cluster\\image-v2-timeseries\\raw_data\\1339_sub"

test_sample_path="C:\\Users\\DELL\\Projects\\MLS_cluster\\image-v2-timeseries\\Encoded_Sample"
test_sample_img="Sample_Subtracted_20040514_20050427"
test_sample_name="Sample_Subtracted_20040514_20050427_raw_data"


#get the paths and names in the dir
# img_path_name,img_f_names = DataProcess.file_name(norm_subtracted_path,".TIF")
# data_path_name,data_f_names = DataProcess.file_name(norm_subtracted_save,".csv")

# DataProcess.img_to_csv(test_sample_path,test_sample_path,test_sample_img)

#transform all the images in the path into csv (done)
# for name in img_f_names:
    # DataProcess.img_to_csv(norm_subtracted_path,norm_subtracted_save,name)


clusteringPara=['BIRCH',50,4,0.5,True]
outlierPara=['LDCOF',4,0.7,2]
runClusteringBased(test_sample_path,test_sample_img,test_sample_path,test_sample_name,test_sample_path\
    ,clusteringPara,outlierPara,'highRank')




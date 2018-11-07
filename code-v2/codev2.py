# from processing import DataProcess as DataProcess
# import clustering
from mypackages.processing import DataProcess
from mypackages.processing import GeoProcess
from mypackages import clustering as cl
from mypackages import clusteringBased as cb
from mypackages import scoresToResults as sc2r

import time

def runClusteringBased(img_path,img_name,data_path,data_name,outlier_save_path,\
    clusteringPara,outlierPara,o_filter="highRank"):
    #clusteringPara[0] is the name, the rest are parameters
    #TODO:change score and filter para
    org_data=DataProcess.csv_to_array(data_path,data_name)
    AlgorithmName=clusteringPara[0]
    print("running "+ AlgorithmName +" for clustering...")
    t0 = time.time()

    if AlgorithmName=="kMeans":
        d_label=cl.kMeans.getCluster(org_data,*(clusteringPara[1]))
    elif AlgorithmName=="Affinity":
        d_label=cl.Affinity.getCluster(org_data,*(clusteringPara[1]))
    elif AlgorithmName=="MeanShift":
        d_label=cl.MeanShift.getCluster(org_data,*(clusteringPara[1]))
    elif AlgorithmName=="Spectral":
        d_label=cl.Spectral.getCluster(org_data,*(clusteringPara[1]))
    elif AlgorithmName=="Agglomerative":
        d_label=cl.Agglomerative.getCluster(org_data,*(clusteringPara[1]))
        AlgorithmName=AlgorithmName+'_'+clusteringPara[1][6]
    elif AlgorithmName=="DBSCAN":
        d_label=cl.DBSCAN.getCluster(org_data,*(clusteringPara[1]))
    elif AlgorithmName=="BIRCH":
        d_label=cl.BIRCH.getCluster(org_data,*(clusteringPara[1]))
    else:
        print("algorithm name ilegal")
        exit()
    AlgorithmName+='_'
    

    t1 = time.time()

    print("running "+ outlierPara[0]+" for calculating the outlier scores...")
    if outlierPara[0]=="LDCOF":
        d_score=cb.calLDCOF.findLDCOF(org_data,d_label,outlierPara[1],outlierPara[2],outlierPara[3])
    
    if o_filter=="highRank":
        outlier_label=sc2r.highRank.getOutliers(d_score,98)

    #save the cluster information
    saveclass_extend_name='_'+AlgorithmName+"cluster_label"
    DataProcess.int_to_csv(outlier_save_path,img_name,d_label,saveclass_extend_name)
    DataProcess.visualize_class(img_path,img_name,outlier_save_path,img_name+saveclass_extend_name)

    #save the label information for further usage
    savelabel_extend_name='_'+AlgorithmName+"outlier_label"
    DataProcess.int_to_csv(outlier_save_path,img_name,outlier_label,savelabel_extend_name)
    GeoProcess.getSHP(img_path,img_name,outlier_save_path,AlgorithmName,outlier_label)#FIXME: the .tif file could not be specified the path
    # DataProcess.visualize_class(img_path,img_name,outlier_save_path,img_name+savelabel_extend_name)
    
    #calculate the Silhouette Coefficient as a reference of the performance of the outcome
    #NOTE:due to the limited memory, I adjust the sample_size to 10000,which may cause the score less reliable
    print("calculating Silhouette Coefficients...")
    clusteringScore=cl.Silhouette.getSilhouette(org_data,d_label,sample_size=10000)
    usingTime=t1-t0
    print("save the information to txt file...")
    with open(data_path+'/'+"runningstatus.txt", 'a') as f:
        f.write("clustering algorithm: "+AlgorithmName+
                "\nsilhouette coefficient: "+ str(clusteringScore)+
                "\nclstering using time: "+ str(usingTime))
        f.write("\n----------------------------------------------\n")
    org_data=None
    return clusteringScore,usingTime

def compareClustering():
    # norm_subtracted_path="C:\\Users\\DELL\\Projects\\MLS_cluster\\image-v2-timeseries\\Encoded_dataset\\Encoded_models_2018-10-03_1339\\subtracted_norm_from_norm"
    # norm_subtracted_save="C:\\Users\\DELL\\Projects\\MLS_cluster\\image-v2-timeseries\\raw_data\\1339_sub"

    test_sample_path="C:\\Users\\DELL\\Projects\\MLS_cluster\\image-v2-timeseries\\Encoded_Sample"
    test_sample_img="Sample_Subtracted_20040514_20050427"
    test_sample_name="Sample_Subtracted_20040514_20050427_raw_data"

    #adjust the parameters here
    clusteringParaS=[
            ["kMeans",[4,5]],#done
            # ["Affinity",[0.5,200,15,True,None,'euclidean',False]], #FIXME: memoryerror, google result: too consuming
            # ["MeanShift",[None,None,False,1,True,None]],#FIXME:a little slow
            # ["Spectral",[4,"discretize",0]],#FIXME: memoryerror, google result: too consuming, need to do calculation previously
            # ["Spectral",[4,"kmeans",0]],    
            ['BIRCH',[50,4,0.5,True]],#done
            # ["DBSCAN"],#too comsuming
            # ["Agglomerative",[4,'euclidean',None,None,'auto','ward','deprecated']],#FIXME: memoryerror
            # ["Agglomerative",[4,'euclidean',None,None,'auto','average','deprecated']],
            # ["Agglomerative",[4,'euclidean',None,None,'auto','complete','deprecated']]
            #TODO: GussianMixture missing
            ]
    outlierPara=['LDCOF',4,0.7,2]

    #run algorithms at the same time
    for clusteringPara in clusteringParaS:
        runClusteringBased(test_sample_path,test_sample_img,
                test_sample_path,test_sample_name,test_sample_path
                ,clusteringPara,outlierPara,'highRank')

compareClustering()


#get the paths and names in the dir
# img_path_name,img_f_names = DataProcess.file_name(norm_subtracted_path,".TIF")
# data_path_name,data_f_names = DataProcess.file_name(norm_subtracted_save,".csv")

# DataProcess.img_to_csv(test_sample_path,test_sample_path,test_sample_img)

#transform all the images in the path into csv (done)
# for name in img_f_names:
    # DataProcess.img_to_csv(norm_subtracted_path,norm_subtracted_save,name)




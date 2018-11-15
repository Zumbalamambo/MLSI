from mypackages.processing import DataProcess
from mypackages.processing import GeoProcess
from mypackages import clustering as cl
from mypackages import clusteringBased as cb
from mypackages import scoresToResults as sc2r
from mypackages.processing import open_image as oi

import numpy as np

norm_img_path="C:\\Users\\DELL\\Projects\\MLS_cluster\\image-v2-timeseries\\Encoded_dataset\\Encoded_models_2018-10-03_1337\\subtracted_norm_from_norm"
norm_data_path="C:\\Users\\DELL\\Projects\\MLS_cluster\\image-v2-timeseries\\raw_data\\1337_sub"
img="Subtracted_20040514_20050427"
raw="Subtracted_20040514_20050427_raw_data"

#get the paths and names in the dir
# img_path_name,img_f_names = DataProcess.file_name(norm_subtracted_path,".TIF")
# data_path_name,data_f_names = DataProcess.file_name(norm_subtracted_save,".csv")

DataProcess.img_to_csv(norm_img_path,norm_data_path,img)

#transform all the images in the path into csv (done)
# for name in img_f_names:
    # DataProcess.img_to_csv(norm_subtracted_path,norm_subtracted_save,name)
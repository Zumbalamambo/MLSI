from osgeo import gdal,gdal_array
import create as cr
import numpy as np 
import open_image as oi 
import array_trs as art
import sys
import os

def img_subtract():
    img_2004=oi.open_tiff("C:\\Users\\DELL\\Projects\\MLS_cluster","Encoded_20040514")
    img_2002=oi.open_tiff("C:\\Users\\DELL\\Projects\\MLS_cluster","Encoded_20021005")

    sub_array=np.subtract(img_2004[0],img_2002[0],dtype="float32")

    dats = cr.create_tiff(nb_channels=img_2004[3],new_tiff_name="subtracted_v1.tif",\
        width = img_2004[2],height= img_2004[1],data_array=sub_array,datatype=gdal.GDT_Float32, \
        geotransformation=img_2004[4],projection=img_2004[5])

    dats=None

# def img_sample():
#     img_2004=oi.open_tiff("C:\\Users\\DELL\\Projects\\MLS_cluster","Encoded_20040514")
#     img_st=oi.open_tiff("C:\\Users\\DELL\\Projects\\MLS_cluster","subtracted_v1")
#     data=img_st[0]
#     sample=data[:][170:850][800:1400]
#     dats = cr.create_tiff(nb_channels=img_st[3],new_tiff_name="subtracted_v1_sample.tif",\
#         width = 600,height= 680,data_array=sample,datatype=gdal.GDT_Float32, \
#          geotransformation=img_2004[4],projection=img_2004[5])#FIXME: projection wrong?
#     # sample=img_st[0][]
#     dats=None
def get_raw_csv(dirpath,savepath,file_name):
    img=oi.open_tiff(dirpath,file_name)# the former one end without "\\"
    data=art.tif2vec(img[0])
    np.savetxt( savepath+"/"+file_name+"_raw_data"+".csv",data,fmt="%.6f",delimiter=';',header="Band-1, Band-2, Band-3")

def csv_to_array(file_path,load_name,extract_name):
    a=np.loadtxt(file_path+"/"+load_name,delimiter=';')
    score=a[:]
    score=score.astype(np.float)
    if extract_name != "NULL":
        np.savetxt(extract_name,score,fmt='%.8f')
    return score



# get_raw_csv("subtracted_v1_sample")
# csv_to_array("1_LOF_sample.csv","1_LOF_score_sample.txt")
# csv_to_array("1_LOF.csv","1_LOF_score_sample.txt",0)
def file_name(file_dir,extendtion):   
    L=[]
    names_no_etd=[]
    for dirpath, dirnames, filenames in os.walk(file_dir):  
        for file in filenames :  
            if os.path.splitext(file)[1] == extendtion:  
                L.append(os.path.join(dirpath, file))
                names_no_etd.append(os.path.splitext(file)[0])
    return L,names_no_etd
#"C:\\Users\\DELL\\Projects\\MLS_cluster\\scores\\"

# def main():
#     if "-getcsv" in sys.argv:
#         get_raw_csv(sys.argv[2])
#     if "-2array" in sys.argv: # MEANS sys.argv[1]=="
#         for i in range(3,len(sys.argv)):
#             csv_to_array(sys.argv[2],sys.argv[i]+".csv",sys.argv[i]+"_sample.txt")
#     if "-f_name" in sys.argv:
#         print(file_name("C:\\Users\\DELL\\Projects\\MLS_cluster\\scores"))

# if __name__ == "__main__":
#     main()
norm_subtracted_path="C:\\Users\\DELL\\Projects\\MLS_cluster\\image-v2-timeseries\\Encoded_dataset\\Encoded_models_2018-10-03_1339\\subtracted_norm_from_norm"
norm_subtracted_save="C:\\Users\\DELL\\Projects\\MLS_cluster\\image-v2-timeseries\\raw_data\\1339_sub"
path_name,f_names = file_name(norm_subtracted_path,".TIF")
for image_name in f_names:
    get_raw_csv(norm_subtracted_path,norm_subtracted_save,image_name)
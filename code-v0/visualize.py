import matplotlib.pyplot as plt
import numpy as np 
import open_image as oi

def visualize_class(file_path,load_name,isSample=True):
    if isSample:
        img=oi.open_tiff("C:\\Users\\DELL\\Projects\\MLS_cluster\\image-v1\\","Subtracted_v1"+"_sample")
    else:
        img=oi.open_tiff("C:\\Users\\DELL\\Projects\\MLS_cluster\\image-v1\\","Subtracted_v1")
    H,W=img[1],img[2]
    data_class=np.loadtxt(file_path+"/"+load_name+".csv")#,delimiter=';'
    data_class=data_class.astype(np.int)
    data_class=data_class.reshape((H,W))
    # print(H,W,data_class.shape)
    
    fig = plt.figure()
    plt.imshow(data_class)
    # plt.show()
    fig.savefig(load_name+'_class_visualization.png')

visualize_class("C:\\Users\\DELL\\Projects\\MLS_cluster", "2_CMGOS_class_sample")
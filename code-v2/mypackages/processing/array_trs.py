import numpy as np
import sklearn
# transform the image_array from <n-band,n-hight,n-width> to <n-hight*n-width,n-band>
def tif2vec(image_array):
    transform1=image_array.transpose(1,2,0)
    flattened_array=transform1.reshape(\
        image_array.shape[1]*image_array.shape[2],\
        image_array.shape[0])
    return flattened_array

#reduce the percision for the squeezer algorithm
def percision_trans(data_array,trans_type,para):
    if trans_type=="digit_int":
        data_array*=para
        r=data_array.astype(int)
    if trans_type=="digit_float":
        r=data_array.astype(float)
        r/=para
    return r

#only use before plt.imshow()
def narray_up_down(vecs,H,W):
    return vecs.reshape(H,W)[::-1]

def data_normalize(data):
    #计算原始数据每行和每列的均值和方差，data是多维数据
    scaler = sklearn. preprocessing.StandardScaler().fit(data)
    #得到每列的平均值,是一维数组
    # mean = scaler.mean_
    #标准化数据
    data_nomal = scaler.transform(data)
    # print("the normalized data:")
    # print(data_nomal[:100])
    return data_nomal


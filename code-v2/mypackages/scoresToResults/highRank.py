import numpy as np

def getOutliers(score_data,percent_boundary):
    b=np.percentile(score_data,percent_boundary)
    print("simply regard the pixels with high rank as outliers")
    print("the value of percent bound is",b)
    result_array=np.zeros(shape=score_data.size)
    result_array[np.where(score_data>b)[0]]=1
    return result_array
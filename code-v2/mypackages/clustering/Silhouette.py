from sklearn import metrics
import numpy as np

def getSilhouette(datas,labels,metric='euclidean', sample_size=None, random_state=None):
    r=metrics.silhouette_score(datas, labels, metric,sample_size,random_state)
    return r
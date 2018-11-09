#just try a hierarchical

from sklearn.cluster import Birch
import numpy as np

def getCluster(org_data,threshold=0.5, branching_factor=50, n_clusters=3, compute_labels=True,copy=True):
	
	brc = Birch(threshold, branching_factor, n_clusters, compute_labels, copy)
	org_data=np.array(org_data)
	# print(org_data)
	brc.fit(org_data)
	d_labels=brc.predict(org_data)
	
	return d_labels
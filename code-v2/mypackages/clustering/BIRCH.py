#just try a hierarchical

from sklearn.cluster import Birch
import numpy as np

def getCluster(org_data,branching_factor =50,n_clusters=4,threshold=0.5,compute_labels=True):
	
	brc = Birch(branching_factor=branching_factor,
			n_clusters=n_clusters, threshold=threshold,
			compute_labels=compute_labels)
	org_data=np.array(org_data)
	# print(org_data)
	brc.fit(org_data)
	d_labels=brc.predict(org_data)
	
	return d_labels
#just try a hierarchical

from sklearn.cluster import Birch

def getCluster(org_data,branching_factor=50,n_clusters=4,threshold=0.5,compute_labels=True):
	brc = Birch(branching_factor=50, n_clusters=None, threshold=0.5,
	compute_labels=True)
	brc.fit(org_data)
	d_labels=brc.predict(X)
	
	return d_labels
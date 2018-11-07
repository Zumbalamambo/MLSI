from sklearn import cluster,mixture


#TODO: figure it out how to use it
def getCluster(X,n_components=4, 
            covariance_type='full', tol=0.001, 
            reg_covar=1e-06, max_iter=100, n_init=1, 
            init_params='kmeans', weights_init=None, 
            means_init=None, precisions_init=None, 
            random_state=None, warm_start=False, 
            verbose=0, verbose_interval=10):
    
    clustering=mixture.GaussianMixture(n_components, 
            covariance_type, tol, 
            reg_covar, max_iter, n_init, 
            init_params, weights_init, 
            means_init, precisions_init, 
            random_state, warm_start, 
            verbose, verbose_interval).fit(X)
    clustering
    

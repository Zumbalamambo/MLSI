class ClusterStructure:

    def __init__(self,nb_features,cluster_type="undefined"):
        # how many features in one member
        self._nb_features=nb_features
        # "large","small" or "undefined"
        self._cluster_type=cluster_type
        #
        self._size=0
        # the tuple to store the pairs of the information of the cluster
        self._summary=list()
        for i in range(nb_features):
            self._summary.append(dict())
        # a tuple to store the id of the member in dataset
        self._tuples=list()

    def change_type(self,c_type):
        self._cluster_type=c_type
    # when add a member, we need to update the summary
    def mem_append(self,tid,member,calSummary=True):
        try:
            self._size+=1
            self._tuples.append(tid)
            if calSummary==True:
                for i in range(member.size):
                    val=member[i]
                    if val in self._summary[i]:
                        self._summary[i][val]+=1
                    else:
                        self._summary[i][val]=1
        except RuntimeError:
            print("the number of features in the member may not be consistent to nb_features")
    
    def simComputation(self,member):
        sim=0
        for i in range(self._nb_features):
            current_feature=self._summary[i]
            if member[i] in current_feature:
                numerator = current_feature[member[i]]
            else:
                continue
            cluster_support=len(self._tuples)
            sim+=numerator/cluster_support
        return sim

    @property
    def nb_features(self):
        return self._nb_features
    @property
    def cluster_type(self):
        return self._cluster_type
    @property
    def tuples(self):
        return self._tuples
    @property
    def size(self):
        return self._size
    @property
    def summary(self):
        return self._summary 

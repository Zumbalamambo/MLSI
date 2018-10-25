import time
import numpy as np
import pprint as pp
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

def squeezer(data_array,nb_features,threshold):
    clusters_=list()
    data_labels=list()#np.array([])#TODO: if want sort, this will change
    t0=time.time()
    for i in range(len(data_array)):
        if i%500000==0:#FIXME:
            print("dealing NO.%d item by squeezer algorithm, running time: %ds" %(i,time.time()-t0))
            print("there are total %d clusters" %(len(clusters_)))
            if i>100:
                print(clusters_[-1].summary)
        member=data_array[i]
        if i==0:
            clusters_.append(ClusterStructure())
            clusters_[0].mem_append(i,member)
            data_labels.append(0)
            # np.append(data_labels,[0])#TODO: if want sort, this will change
        else:
            sim_index=0
            biggest_sim=0
            for j in range(len(clusters_)):
                temp_sim=clusters_[j].simComputation(member)
                
                # if i%1000==0:#FIXME:
                    # print("the biggest_sim now is %d, the sim_index is %d, the temp_sim is %d"\
                        #  %(biggest_sim, sim_index, temp_sim))
                if biggest_sim < temp_sim :
                    biggest_sim=temp_sim
                    sim_index=j
            if biggest_sim>=threshold:
                clusters_[sim_index].mem_append(i,member)
                data_labels.append(sim_index)
                # np.append(data_labels,[sim_index])#TODO: if want sort, this will change
            else:
                clusters_.append(ClusterStructure(nb_features))
                clusters_[-1].mem_append(i,member)
                data_labels.append(len(clusters_)-1)
                # np.append(data_labels,[len(clusters_)-1])#TODO: if want sort, this will change
        # if i%1000==0:
            # pp.pprint(data_labels[-1000:-1])
    data_labels_array=np.array(data_labels)
    return data_labels_array,clusters_

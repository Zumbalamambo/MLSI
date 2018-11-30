import os
import sys
import time

from sklearn.utils import check_X_y
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from mypackages.processing import DataProcess
from mypackages.processing import GeoProcess
from mypackages.processing import open_image as oi
from mypackages.processing import array_trs as art
from mypackages.scoresToResults import highRank

import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.utils.data as Data

import numpy as np
# NOTE: do with whole original image
# anomaly detection pipeline parameters
# network_name='no_extract_net.pkl'
network_name='full_train_net.pkl'
# 1.traing phase
trainMode=True
# 2.prediction phase
# trainMode=False


# training parameters
batch_size=1
epochs=20
log_interval=20
torch.manual_seed(10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(5)

class tiffData():
    
    def __init__(self, isExtract=True,
        root_dir="C:\\Users\\DELL\\Projects\\VHR_CD\\image-v2-timeseries\\newest",
        filename="4Band_Subtracted_20040514_20050427"):
        """
        Args:
            root_dir (string): the path of the file
            file_name (string): the name of the picture
            isExtract (boolean): use the mask to extract changed area
        """
        self.isExtract=isExtract
        self.root_dir = root_dir
        self.file_name= filename
        self.dataset=oi.open_tiff(root_dir,filename)
        self.H=self.dataset[1]
        self.W=self.dataset[2]
        self.n_bands=self.dataset[3]
        self.npdataset=art.tif2vec(self.dataset[0])#flatten and transform the array
        

        if self.isExtract:
            # extract out the changed area
            self.select_path="C:\\Users\\DELL\\Projects\\VHR_CD\\image-v2-timeseries\\EXTRACT"
            self.select_img="SOMOCLU_20_20_HDBSCAN_cl_2_2004_2005_min_cluster_size_4_alg_best_"
            self.simg=oi.open_tiff(self.select_path,self.select_img)
            self.select=self.simg[0]#(2720000)        
            self.changePos=DataProcess.selectArea(self.select,self.n_bands,-1,isStack=True)
            self.ns_changePos=DataProcess.selectArea(self.select,self.n_bands,-1,isStack=False)
            self.ns_nonChangePos=DataProcess.selectArea(self.select,self.n_bands,0,isStack=False)

            self.npdataset=self.npdataset[self.changePos].reshape(-1,self.n_bands)


        # normalization
        self.nmax=self.npdataset.max(axis=0)
        self.nmin=self.npdataset.min(axis=0)
        self.norm_data = (self.npdataset -self.nmin)/(self.nmax-self.nmin)

        # TODO:don't know what's for yet, add only to be compatible to TensorDataset
        self.target_data=np.zeros_like(self.norm_data)

        #clear the memory
        # self.simg=None
        self.dataset=None


    # must rewritten method

    def __len__(self):
        return self.norm_data.shape[0]
    def __getitem__(self, idx):
        return self.norm_data[idx]

    def getData(self):
        return self.norm_data

    def getTargetData(self):
        return self.target_data

    def getH(self):
        return self.H
    def getW(self):
        return self.W
    def getBand(self):
        return self.n_bands
    
    # TODO:
    def anomaly(self,scores):
        if isExtract:
            self.score_result=np.empty_like(self.select.reshape(-1,1))
            # scale the scores
            self.score_result[self.ns_changePos]=DataProcess.scaleNormalize(scores,(0,500)).reshape(-1,)
            self.score_result[self.ns_nonChangePos]=0
        else:
            self.score_result=DataProcess.scaleNormalize(scores,(0,500)).reshape(-1,)
        # give labels
        self.outlier_result=highRank.getOutliers(self.score_result,99)
        # generate picture
        GeoProcess.getSHP(img_path=self.root_dir,img_name=self.file_name,
            save_path="C:\\Users\\DELL\\Projects\\VHR_CD\\repository\\code-v2",extend_name="VAE_noEXT_",result_array=self.outlier_result)

# import the data
x=tiffData(isExtract=isExtract)
my_dataset=Data.TensorDataset( torch.from_numpy(x.getData()).float(),
    torch.from_numpy(x.getTargetData()).float())

if trainMode:
    train_loader=Data.DataLoader(my_dataset, batch_size=batch_size, 
        shuffle=trainMode)#no need to shuffle when evaluation
else:
    train_loader=Data.DataLoader(my_dataset, batch_size=1, 
        shuffle=trainMode)#no need to shuffle when evaluation

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(4, 10) # input data
        # TODO: add one more layer for relu
        self.fc21 = nn.Linear(10, 2) # get mu
        self.fc22 = nn.Linear(10, 2) # get variance
        self.fc3 = nn.Linear(2, 10)
        # TODO: add one more layer for relu
        self.fc4 = nn.Linear(10, 4)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    # Assume the posterior q(z|x) is normal distribution
    # and since sampling itself can not be done gradient
    # so we do the gradient descent on the result of sampling.
    # Notice that: 
    # sample Z from N(mu,var) <=> sample eps from N(0,1) and let Z=mu+eps*var
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        # NOTE: to see the size of z
        # print(std.cpu().data.numpy().shape )
        # exit()
        #torch.randn() by default is N(0,1)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        return z

    def decode(self, z):
        h3 = F.relu(self.fc3(z))

        output=torch.sigmoid(self.fc4(h3))
        return output

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 4))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 4), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD
# only return the reconstruction probability
def rcon_prob_BCE(recon_x, x, mu, logvar):

    return F.binary_cross_entropy(recon_x, x.view(-1,4), reduction='sum')

# create the network
if trainMode:
    model = VAE().to(device)
    # decide how to update the weights, learning rate default=1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-1)
else:
    model=torch.load(network_name,map_location=lambda storage, loc: storage)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data,_) in enumerate(train_loader):
        # move the data to gpu if possible
        data = data.to(device)
        # clear the gradient
        optimizer.zero_grad()
        # input batch data and obtain output
        # <class 'torch.Tensor'> <class 'torch.Tensor'> <class 'torch.Tensor'>
        # grad_fn=<SigmoidBackward> grad_fn=<ThAddmmBackward> grad_fn=<ThAddmmBackward>
        recon_batch, mu, logvar = model(data)
        # claculate the loss <class 'torch.Tensor'>
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        # update the parameters in the network
        optimizer.step()
        # dynamically print out the loss according to the log_interval
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def evaluation():
    model.eval()
    np_recon=np.empty(shape=(x.__len__(),))
    with torch.no_grad():
        t0=time.clock()
        for idx in range(x.__len__()):
            # # move the data to gpu if possible
            # TODO: 如果还是不行，就试一下这一句，因为现在是model在CPU上了已经 ->已解决
            # data = data.to(device)
            data=torch.from_numpy(x.__getitem__(idx)).float()
            recon_batch, mu, logvar = model(data)
            # print(recon_batch,x)
            # claculate the loss
            recon_prob= rcon_prob_BCE(recon_batch, data, mu, logvar)
            np_recon[idx]=recon_prob
            if idx % 100000 ==0:
                print("done",idx,"predictions, evaluating using time:",time.clock()-t0)
                
            # print(type(recon_prob))
            # print(recon_prob)
            # if idx==100:
            #     break
    x.anomaly(np_recon)
            
        
        # for i, (data, _) in enumerate(train_loader):
        #     # # move the data to gpu if possible
        #     # data = data.to(device)
        #     data=data.to(device)
        #     recon_batch, mu, logvar = model(data)
        #     # print(recon_batch,x)
        #     # claculate the loss
        #     recon_prob= rcon_prob_BCE(recon_batch, data, mu, logvar)
        #     # print(type(recon_prob))
        #     # print(recon_prob)
        #     if i==100:
        #         break

    print("eval time:",time.clock()-t0)
    # NOTE:
    # on GPU: eval time for 100 point 0.454549907970334
    # on CPU: eval time for 100 point 0.02499042962584874
    # choose CPU obviously

if __name__ == '__main__':
    '''
    TODO:如果说，generative model的目的是学习正常的模式，那么是不是应该不做extraction？
    NOTE: no extraction 的记录
    '''
    # print to test what we load
    # for i in range(300,310):
    #     print("get an item:",train_loader.dataset[i])
    
    # train the model
    if trainMode:
        for epoch in range(1, epochs + 1):
            train(epoch)
        # save the network we trained
        torch.save(model,network_name)
    else:
        # TODO:calcuate the reconstruction probability
        evaluation()
        
        # TODO:label out the anomaly and get the outlier picture




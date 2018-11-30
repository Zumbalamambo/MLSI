import os
import sys
import time

from osgeo import gdal
from mypackages.processing import DataProcess
from mypackages.processing import GeoProcess
from mypackages.processing import create
from mypackages.processing import open_image as oi
from mypackages.scoresToResults import highRank

import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.utils.data as Data
import numpy as np
from visdom import Visdom

viz=Visdom(env='VAE')
assert viz.check_connection()
viz.text('Visdom is starting', win='text1')
class log():
    def __init__(self):
        self.CSTEP=0
        self.count_log=[0]
        self.loss_log=[]
    def inc(self):
        self.CSTEP+=1
        self.count_log.append(self.CSTEP)
    def getC(self,i):
        return self.count_log[i]
    def getL(self,i):
        return self.loss_log[i]
    def addL(self,l):
        self.loss_log.append(l)

vlog=log()

network_name='0504_conv_VAE_net.pkl'
trainMode=not True # NOTE: switch mode before run
# training parameters
slice_d=10
batch_size=100
epochs=50
log_interval=20
torch.manual_seed(10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(5)

class tiffData():
    
    def __init__(self,filename,isExtract=True,
        root_dir="C:\\Users\\DELL\\Projects\\VHR_CD\\image-v2-timeseries\\Montpellier_SPOT5_Clipped_relatively_normalized_03_02_mask1"
        ):
        """
        Args:
            root_dir (string): the path of the file
            file_name (string): the name of the picture
        """
        self.root_dir = root_dir
        self.file_name= filename
        self.dataset=oi.open_tiff(root_dir,filename)
        self.H=self.dataset[1] #1700
        self.W=self.dataset[2] #1600
        self.n_bands=self.dataset[3]
        self.geo=self.dataset[4]
        self.prj=self.dataset[5]
        
        self.norm_data=self.normalization()
        # TODO:don't know what's for yet, add only to be compatible to TensorDataset
        self.target_data=np.zeros_like(self.norm_data)

        #clear the memory
        self.dataset=None
    def __len__(self):
        length=self.norm_data.shape[0]# NOTE: Tensor.size() is lazy
        return length
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
    def normalization(self):
        # change dimension
        ndata=self.dataset[0].transpose(1,2,0)
        # normalization
        tM=ndata.reshape(-1,self.n_bands)
        tmin=np.min(tM,axis=0)
        tmax=np.max(tM,axis=0)
        tM=(tM-tmin)/(tmax-tmin)
        ndata=tM.reshape(self.H,self.W,self.n_bands)
        # slice batches
        # len(x)==27200, x[0].size()==[10,10,4]
        x=list()
        a=np.array_split(ndata,self.H/slice_d,0)
        for i in a:
            b=np.array_split(i,self.W/slice_d,1)
            for j in b:
                x.append(j)
        # concatenate
        norm_data=np.stack(x,0)
        return norm_data.transpose(0,3,1,2)
    def recon_img(self,reconbatches,extend_name):
        # len(reconbatches)=272, reconbatches[0].size()=(100,4,10,10)
        # reconbatches[0].min()=-1.9 reconbatches[0].max()= -1.0

        reconbatches=np.concatenate(reconbatches,axis=0)#=>27200,4,10,10         
        reconbatches=reconbatches.transpose(0,2,3,1)#=> 27200,10,10,4
        rs1=np.array_split(reconbatches,self.H/slice_d,axis=0)
        # rec_data=np.stack(rows,axis=1)
        rs2=[]# => 170 * (10,1600,4)
        for r in rs1:
            rs2.append(np.concatenate(r,axis=1))
        rec_data=np.concatenate(rs2)

        # tM=rec_data.reshape(-1,self.n_bands)
        # tmin=np.min(tM,axis=0)
        # tmax=np.max(tM,axis=0)
        # tM=500*(tM-tmin)/(tmax-tmin)
        # rec_data=tM.reshape(self.H,self.W,self.n_bands)
        rec_data=rec_data.transpose(2,0,1)
        print(rec_data)
        # generate picture
        create.create_tiff(nb_channels=4,new_tiff_name=extend_name+"VAEreconstruct.tif",
        width=self.W,height=self.H,datatype=gdal.GDT_Float64,data_array=rec_data,
        geotransformation=self.geo,projection=self.prj)
        # GeoProcess.getTIF(img_path=self.root_dir,img_name=self.file_name,
        #     save_path="C:\\Users\\DELL\\Projects\\VHR_CD\\repository\\code-v2",
        #     extend_name="VAE_re_",result_array=rec_data)

# import the data, train with 2 images
x1=tiffData("SPOT5_HRG1_XS_20040514_N1_SCENE_047_262") #img1
y1=Data.TensorDataset( torch.from_numpy(x1.getData()).float(),
torch.from_numpy(x1.getTargetData()).float())
loader1=Data.DataLoader(y1, batch_size=batch_size, 
    shuffle=False)#no need to shuffle when evaluation
x2=tiffData("SPOT5_HRG2_XS_20050427_N1_SCENE_047_262_0") #img2
y2=Data.TensorDataset( torch.from_numpy(x2.getData()).float(),
torch.from_numpy(x2.getTargetData()).float())
loader2=Data.DataLoader(y2, batch_size=batch_size, 
    shuffle=False)#no need to shuffle when evaluation


class VAE(nn.Module): 
    def __init__(self):
        super(VAE, self).__init__()
        #padding=(kernel_size-1)/2 当 stride=1
        self.conv1=nn.Sequential(
            nn.Conv2d( in_channels=4, out_channels=8,
                kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(8*100,200) # input data
        # TODO: add one more layer for relu
        self.fc21 = nn.Linear(200, 30) # get mu
        self.fc22 = nn.Linear(200, 30) # get variance
        self.fc3 = nn.Linear(30, 200)
        # TODO: add one more layer for relu
        self.fc4 = nn.Linear(200,8*100)
        self.dconv1=nn.ConvTranspose2d(in_channels=8,out_channels=4,
            kernel_size=3,stride=1,padding=1)
        
        #initialize parameters
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

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
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        return z

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        # TODO: try other activation functions
        output=torch.relu(self.fc4(h3))
        return output

    def forward(self, x):
        hx=self.conv1(x)
        mu, logvar = self.encode(hx.view(-1, 8*slice_d*slice_d))
        z = self.reparameterize(mu, logvar)
        hx_t=self.decode(z)#, mu, logvar
        output=self.dconv1(hx_t.view(batch_size,8,slice_d,slice_d),output_size=(batch_size,4,slice_d,slice_d))
        return output,mu,logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    getBCE=nn.BCEWithLogitsLoss(reduction='sum')
    BCE = getBCE(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
# only return the reconstruction probability
def rcon_prob_BCE(recon_x, x, mu, logvar):
    neg_abs = - recon_x.abs()
    BCE=recon_x.clamp(min=0) - recon_x * x + (1 + neg_abs.exp()).log()
    return BCE.mean()

# create the network
if trainMode :
    # NOTE: if want to load, copy from beneath
    model = VAE().to(device)
    # decide how to update the weights, learning rate default=1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-5)#,weight_decay=1e-3
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=2,gamma=0.9)
else:
    model=torch.load(network_name,map_location=lambda storage, loc: storage)



def train(epoch,train_loader):
    model.train()
    train_loss = 0
    for step, (data,_) in enumerate(train_loader):
        data = data.to(device)# move the data to gpu if possible
        optimizer.zero_grad()# clear the gradient
        # input batch data and obtain output
        # <class 'torch.Tensor'> <class 'torch.Tensor'> <class 'torch.Tensor'>
        # grad_fn=<SigmoidBackward> grad_fn=<ThAddmmBackward> grad_fn=<ThAddmmBackward>
        recon_batch, mu, logvar = model(data)
        # print(recon_batch.size(),data.size())
        # claculate the loss <class 'torch.Tensor'>
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        # update the parameters in the network
        optimizer.step()
        # dynamically print out the loss according to the log_interval
        if step % log_interval == 0:
            slog1='Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(data), len(train_loader.dataset),
                100. * step / len(train_loader),
                loss.item() / len(data))
            print(slog1)
            viz.text(slog1, win='text1', append=True)
        #NOTE: use to visualize
        # vlog.addL(loss.item() / len(data))
        # vlog.inc()
    scheduler.step()
    slog2='====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset))
    print(slog2)
    viz.text(slog2, win='text1', append=True)
    vlog.addL(train_loss / len(train_loader.dataset))
    vlog.inc()
    for i in range(vlog.CSTEP-1,vlog.CSTEP):  
        viz.line(X=torch.FloatTensor([vlog.getC(i)]), Y=torch.FloatTensor([vlog.getL(i)]), win='loss', update='append' if i>0 else None)
    return loss.item() / len(data)

def generate(train_loader,x,IMG):
    model.eval()
    model.to(device)
    recon=[]
    t0=time.clock()
    with torch.no_grad():
        for batch_idx, (data,_) in enumerate(train_loader):
            data = data.to(device)# move the data to gpu if possible
            recon_batch, mu, logvar = model(data)
            recon.append(recon_batch)

    x.recon_img(recon,IMG+"_")

    print("reconstruction time:",time.clock()-t0)


if __name__ == '__main__':
    '''
    TODO:如果说，generative model的目的是学习正常的模式，那么是不是应该不做extraction？
    NOTE: no extraction 的记录
    '''
    # print to test what we load
    # for i in range(300,310):
    #     print("get an item:",train_loader.dataset[i])
    
    # train the model
    losslist=[]
    temploss=300
    if trainMode:
        for epoch in range(1, epochs + 1):
            l=train(epoch,loader2) 
            if temploss-l >0:
                print("-----loss decreasing-----")
                if temploss-l < 1e-2:
                    print("-----loss in convergence-----")
            else:
                print("!!! loss increasing +")
            losslist.append(l)
            temploss=l
        for epoch in range(1, epochs + 1):
            l=train(epoch,loader1) 
            if temploss-l >0:
                print("-----loss decreasing-----")
                if temploss-l < 1e-2:
                    print("-----loss in convergence-----")
            else:
                print("!!! loss increasing +")
            losslist.append(l)
            temploss=l
        # save the network we trained
        torch.save(model,network_name)

    else:
        # TODO:calcuate the reconstruction probability
        generate(loader1,x1,"2004")
        generate(loader2,x2,"2005")
        
        # TODO:label out the anomaly and get the outlier picture




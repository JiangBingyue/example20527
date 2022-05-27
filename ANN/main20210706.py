import parser
import torch
from torch import functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
import math
import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from sklearn.model_selection import KFold
import warnings
# from visdom import Visdom

warnings.filterwarnings("ignore")
# viz=Visdom()
# viz.line([[0.,0.]],[0.],win='train_loss',opts=dict(title='train%test loss',
#                                                    legend=['train_loss','test_loss']))

batch_size=20
learning_rate=0.01
epoches=100
original_dim=7
split=5

def calc_mean_std(data):
    raw,col=data.shape
    mean_data=np.zeros(col)
    std_data=np.zeros(col)
    for ii in range(col):
        mean_data[ii]=data[:,ii].mean()
        std_data[ii]=data[:,ii].std()
    return mean_data,std_data
def reverse_nomal(data,mean_data,std_data):
    data = data*std_data + mean_data
    return data

def nomalization(data,mean_data,std_data):
    data=(data-mean_data)/std_data
    return data
def write_csv(data,name):
    file_name = name
    save = pd.DataFrame(list(data))
    try:
        save.to_csv(file_name)
    except UnicodeEncodeError:
        print("编码错误")

class regressDataset(Data.Dataset):
    def __init__(self,data,idx):
        self.len_data = data[idx].shape[0]
        data = torch.from_numpy(data)
        self.x = data[:, :-1].float()
        self.y = data[:, -1].float()
        self.y = self.y.unsqueeze(1)
        self.x=self.x[idx,:]
        self.y=self.y[idx,:]
    def __len__(self):
        return self.len_data
    def __getitem__(self, index):
        return self.x[index],self.y[index]

#载入数据
data=pd.read_csv(os.path.join(os.path.abspath('.\\ANN'),'data.csv'),header=None)
data=np.array(data)
mean_data,std_data=calc_mean_std(data)
data=nomalization(data,mean_data,std_data)
len_data=data.shape[0]
idx_data=np.arange(len_data)
kf=KFold(n_splits=split,shuffle=True)
class ANN(nn.Module):
    def __init__(self,original_dim):
        super(ANN,self).__init__()
        self.original_dim=original_dim
        self.model=nn.Sequential(
            nn.Linear(original_dim,100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 1),
        )
    def forward(self,x):
            x=self.model(x)
            return x

# 定义神经网络
net = ANN(original_dim=original_dim)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
criteon = nn.MSELoss(reduce=True, size_average=True)

#数据预处理
ii=0
mse5Fold=[]
arrayLossTest = []
arrayLossTrain = []
arrayErrorTest = []
arrayErrorTrain = []
logitsPred = np.array(np.zeros(1))
targetsPred = np.array(np.zeros(1))
logitsPredtest = np.array(np.zeros(1))
targetsPredtest = np.array(np.zeros(1))
for train_idx, test_idx in kf.split(idx_data):
    print('fold: %d' % ii)
    ii+=1
    net = ANN(original_dim=original_dim)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criteon = nn.MSELoss(reduce=True, size_average=True)
    train_db = regressDataset(data,train_idx)
    test_db = regressDataset(data,test_idx)
    train_loader = Data.DataLoader(
        dataset=train_db,
        batch_size=batch_size,
        shuffle=True,)
    test_loader = Data.DataLoader(
        dataset=test_db,
        batch_size=batch_size,
        shuffle=True, )
    outfile1 = open('./epoch-loss'+str(ii)+'.txt', 'w')
    minLossTest=1
    minLossTrain=1
    minErrorTest = 1
    minErrorTrain = 1

    for epoch in range(epoches):
        train_loss=0
        trainError = 0
        for batch_idx,(data0,target) in enumerate(train_loader):
            # data=data.view(-1,28*28)
            logits=net(data0)

            loss=criteon(logits,target)
            train_loss+=loss.item()
            trainError += (((reverse_nomal(logits, mean_data[-1], std_data[-1]) -
                            reverse_nomal(target, mean_data[-1], std_data[-1])). \
                           abs() / reverse_nomal(target, mean_data[-1], std_data[-1])).mean()).item()
            if epoch==epoches-1 and ii==1:
                logitsPred=np.vstack((logitsPred, logits.cpu().detach().numpy()))
                targetsPred = np.vstack((targetsPred , target.cpu().detach().numpy()))
            optimizer.zero_grad()#梯度初始化为0
            loss.backward()
            optimizer.step()

            # if batch_idx % 10 ==0:
            #     print('train epoch:{} [{}/{} ({:.0f})]\tLoss:{:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #         100.*batch_idx/len(train_loader),loss.item()))

            test_loss = 0
            testError=0
            precision = 0
        for data0, target in test_loader:
            #data = data.view(-1,28*28)
            logits = net(data0)
            test_loss += criteon(logits, target).item()#item的作用是把Tensor转换成张量
            testError+=(((reverse_nomal(logits,mean_data[-1],std_data[-1])-
                          reverse_nomal(target, mean_data[-1], std_data[-1])).\
                         abs()/reverse_nomal(target, mean_data[-1], std_data[-1])).mean()).item()
            if epoch==epoches-1 and ii==1:
                logitsPredtest = np.vstack((logitsPredtest, logits.cpu().detach().numpy()))
                targetsPredtest = np.vstack((targetsPredtest, target.cpu().detach().numpy()))
        testError /= math.ceil(len(test_loader.dataset) / batch_size)
        test_loss /= math.ceil(len(test_loader.dataset)/batch_size)
        trainError/= math.ceil(len(train_loader.dataset)/batch_size)
        train_loss/= math.ceil(len(train_loader.dataset)/batch_size)

        if minLossTest>test_loss:
            minLossTest=test_loss
        if minLossTrain>train_loss:
            minLossTrain=train_loss
        if minErrorTest>testError:
            minErrorTest=testError
        if minErrorTrain>trainError:
            minErrorTrain=trainError
        # viz.line([[train_loss,test_loss]], [epoch], win='train_loss', update='append')
        outfile1.write(str(epoch) + '  ' + str(train_loss)+ '  ' + str(test_loss) + '\n')
        # print('\nTest set:Average loss:{:.4f},Precision: ({:.0f}%)\n'.format(
        #     test_loss,100.*precision/len(test_loader.dataset)))
    arrayLossTest.append(minLossTest)
    arrayLossTrain.append(minLossTrain)
    arrayErrorTest.append(minErrorTest)
    arrayErrorTrain.append(minErrorTrain)
    outfile1.close()
    break

write_csv(reverse_nomal(logitsPred[1:],mean_data[-1],std_data[-1]),'y_pred.csv')
write_csv(reverse_nomal(targetsPred[1:],mean_data[-1],std_data[-1]), 'y_real.csv')
write_csv(reverse_nomal(logitsPredtest[1:],mean_data[-1],std_data[-1]),'y_predtest.csv')
write_csv(reverse_nomal(targetsPredtest[1:],mean_data[-1],std_data[-1]), 'y_realtest.csv')

print('%.4f(%.4f)' % (np.array(arrayLossTest).mean(),np.array(arrayLossTest).std()))
print('%.4f(%.4f)' % (np.array(arrayLossTrain).mean(), np.array(arrayLossTrain).std()))
print('%.4f(%.4f)' % (np.array(arrayErrorTest).mean(),np.array(arrayErrorTest).std()))
print('%.4f(%.4f)' % (np.array(arrayErrorTrain).mean(),np.array(arrayErrorTrain).std()))

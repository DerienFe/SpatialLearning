import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
import math
import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import time


from scipy import stats
import os

# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#print(torch.cuda.is_available(), device)

'''load input data'''
features = np.load('ki_ligand_features.npy')
adj = np.load('ki_ligand_adj.npy')
length = np.load('ki_length.npy')
activity = np.load('ki_activity.npy')

'''transformation output activity'''

activity = np.log(1/(activity/1e6))
plt.hist(activity,bins=50)
plt.show()

'''normalize adjacency matrix'''
import scipy.sparse as sp


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


for i in range(adj.shape[0]):
    adj[i][:length[i], :length[i]] = adj[i][:length[i], :length[i]] + np.eye(length[i])
    adj[i] = normalize(adj[i])

'''prepare data to torch tensor'''
features = torch.tensor(features)
adj = torch.tensor(adj)
length = torch.tensor(length)
activity = torch.tensor(activity.reshape([activity.shape[0], 1]))
train_x = features[:500]
train_adj = adj[:500]
train_length = length[:500]
train_y=activity[:500]

valid_x = features[500:600]
valid_adj = adj[500:600]
valid_length = length[500:600]
valid_y=activity[500:600]

test_x = features[600:]
test_adj = adj[600:]
test_length = length[600:]
test_y=activity[600:]


scaler = preprocessing.StandardScaler().fit(activity[:500])
train_y=scaler.transform(activity[:500])
valid_y=scaler.transform(activity[500:600])
test_y=scaler.transform(activity[600:])


if torch.cuda.is_available():
    train_x = train_x.to(device)
    train_adj = train_adj.to(device)
    train_length = train_length.to(device)
    train_y = torch.tensor(train_y).to(device)
    valid_x = valid_x.to(device)
    valid_adj = valid_adj.to(device)
    valid_length = valid_length.to(device)
    valid_y = torch.tensor(valid_y).to(device)
    test_x=test_x.to(device)
    test_adj=test_adj.to(device)
    test_length=test_length.to(device)
    test_y=torch.tensor(test_y).to(device)

"""GCN"""

'''GCN layer'''


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        output = torch.zeros(input.shape[0], input.shape[1], self.out_features)
        for i in range(input.shape[0]):
            support = torch.mm(input[i].double(), self.weight.double())
            output[i] = torch.spmm(adj[i], support.double())
            if self.bias is not None:
                for j in range(input.shape[1]):
                    output[i, j] = output[i, j] + self.bias

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class ave_pooling(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self):
        super(ave_pooling, self).__init__()

    def forward(self, input, length):
        output = torch.zeros([input.shape[0], input.shape[-1]])
        for i in range(input.shape[0]):
            output[i] = input[i, :int(length[i]), :].mean(0)

        return output

    def __repr__(self):
        return self.__class__.__name__


'''GCN model'''


class GCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid2)
        self.ave_pooling = ave_pooling()
        self.dropout = dropout
        self.linear = torch.nn.Linear(nhid2, 1)

    def forward(self, x, adj, length):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        # print(x[0,:,0])
        # print(x[0,:int(length[0]),0].mean())
        x = self.ave_pooling(x, length)
        x = self.linear(x)
        # print(y)
        # x=self.linear(y)
        # print(x)
        # print(x)
        #
        #
        # y=self.ave_pooling(x,length)
        # y=torch.zeros(x.shape[0],x.shape[-1])
        # for i in range(x.shape[0]):
        #  y[i]=x[i,:int(length[i]),:].mean(0)
        return x


if torch.cuda.is_available():
    GCN_model = GCN(nfeat=4, nhid1=100, nhid2=50, dropout=0.1).to(device)
else:
    GCN_model = GCN(nfeat=4, nhid1=100, nhid2=50, dropout=0.1)
print(GCN)

"""check which parameter is not on gpu"""
for name, param in GCN_model.named_parameters():
    if param.device.type != 'cuda':
        print('param {}, not on GPU'.format(name))

'''train model'''

torch_dataset = Data.TensorDataset(train_x, train_adj, train_length, train_y)
loader = Data.DataLoader(
    dataset=torch_dataset,  # torch TensorDataset format
    batch_size=32,  # mini batch size
    shuffle=True,
)

optimizer = torch.optim.Adam(GCN_model.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()
'''
start = time.time()
hist_train_loss = []
hist_valid_loss = []
for t in range(50):
    print(t)
    for step, (batch_x, batch_adj, batch_length, batch_y) in enumerate(loader):
        # print('step:',step)
        prediction = GCN(batch_x, batch_adj, batch_length)
        # print(prediction,batch_y)
        loss = loss_func(prediction, batch_y.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    prediction_train = GCN(train_x, train_adj, train_length)
    loss_train = loss_func(prediction_train, train_y.float())
    prediction_valid = GCN(valid_x, valid_adj, valid_length)
    loss_valid = loss_func(prediction_valid, valid_y.float())
    hist_train_loss.append(loss_train.data.cpu().numpy())
    hist_valid_loss.append(loss_valid.data.cpu().numpy())
    print('loss: ', loss_train.data.cpu().numpy(), 'valid_loss:', loss_valid.data.cpu().numpy())
    print(time.time() - start)

print(time.time() - start)
'''
def train(epochs,patient,PATH,earlystopping=True):
    start = time.time()
    hist_train_loss = []
    hist_valid_loss = []
    best_loss_valid = 1e10
    best_t=0
    for t in range(epochs):
        print(t)
        for step, (batch_x, batch_adj, batch_length, batch_y) in enumerate(loader):
            # print('step:',step)
            prediction = GCN_model(batch_x, batch_adj, batch_length)
            # print(prediction,batch_y)
            loss = loss_func(prediction, batch_y.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        prediction_train = GCN_model(train_x, train_adj, train_length)
        loss_train = loss_func(prediction_train, train_y.float())
        prediction_valid = GCN_model(valid_x, valid_adj, valid_length)
        loss_valid = loss_func(prediction_valid, valid_y.float())
        hist_train_loss.append(loss_train.data.cpu().numpy())
        hist_valid_loss.append(loss_valid.data.cpu().numpy())
        print('loss: ', loss_train.data.cpu().numpy(), 'valid_loss:', loss_valid.data.cpu().numpy())
        print(time.time() - start)

        if earlystopping:
            if best_loss_valid>loss_valid:
               best_loss_valid=loss_valid
               best_t=t
               torch.save(GCN_model.state_dict(),PATH)
            if t-best_t>patient:
                break

    print(time.time() - start)
def test(model):
    model.eval()
    prediction=model(test_x,test_adj,test_length)
    loss_test = loss_func(prediction,test_y.float())
    sp_cor=stats.spearmanr(prediction.data.cpu().numpy(),test_y.data.cpu().numpy())[0]
    rsq=metrics.r2_score(test_y.data.cpu().numpy(),prediction.data.cpu().numpy())
    print("test set results:",
          "loss= ",loss_test.data.cpu().numpy(),
          "spearman correlation=", sp_cor,
          "r squared=", rsq)


train(epochs=150,patient=20,PATH='GCN.pth')
GCN_model = GCN(nfeat=4, nhid1=100, nhid2=50, dropout=0.1).to(device)
GCN_model.load_state_dict(torch.load('GCN_ki_ligand.pth',map_location=torch.device('cpu')))
test(GCN_model)

"""result plot"""
'''
prediction=GCN_model(test_x,test_adj,test_length)
plt.plot(prediction.data.cpu().numpy(),test_y.data.cpu().numpy(),'.')
plt.ylabel('true values')
plt.xlabel('predicted values')

plt.hist(prediction.data.cpu().numpy(),bins=50,alpha=0.5,label='predicted')
plt.hist(test_y.data.cpu().numpy(),bins=50,alpha=0.5,label='true')
plt.legend(loc='upper right')
plt.show()
'''

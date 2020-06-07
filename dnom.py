from datetime import datetime
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn import preprocessing
import scipy.io as sio
import util


class MLP(torch.nn.Module):
    
    def __init__(self, i, h, o):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(i, h)
        self.bn = torch.nn.BatchNorm1d(h)
        self.act = torch.nn.LeakyReLU(True)
        self.fc2 = torch.nn.Linear(h, o)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class DNOM(object):
    
    def __init__(self, i, h, o, B):
        self.model = MLP(i, h, o).cuda()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.00005)
        self.B = B

    def train(self, x):
        self.model.train()
        
        o = self.model(x)
        loss = torch.mean(torch.pow(torch.mm(o, self.B.t()) - x, 2))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        U, _, V = torch.svd(torch.mm(x.t().data, o.data))
        self.B = torch.autograd.Variable(torch.mm(U, V.t()))
        
        return loss.data.cpu().numpy()

    def predict(self, x):
        self.model.eval()
        out = self.model(x)
        return out


def _hstack(data, n_samples):
    length = data.shape[0]
    result = []
    for i in range(n_samples):
        result.append(data[i: i + length - n_samples + 1])
    return np.hstack(result)


def get_data(n_samples):
    data, _ = util.read_data(error=0, is_train=True)
    train_data = _hstack(data, n_samples)

    test_data = []
    for i in range(22):
        data, _ = util.read_data(error=i, is_train=False)
        test_data.append(_hstack(data, n_samples))
    test_data = np.vstack(test_data)

    scaler = preprocessing.StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    return train_data, test_data


def main():
    train_data, test_data = get_data(n_samples=3)
    pca = PCA(52 * 3).fit(train_data)
    B = torch.autograd.Variable(torch.from_numpy(pca.components_.T).cuda())
    
    x = torch.autograd.Variable(torch.from_numpy(train_data).cuda())

    nca = DNOM(52*3, 52*3, 52*3, B)

    for i in range(500):
        loss = nca.train(x)
        if i % 5 == 0:
            print('{}  epoch[{}]  loss = {:0.3f}'.format(datetime.now(), i, loss[0]))
    
    pred = nca.predict(torch.autograd.Variable(torch.from_numpy(test_data).cuda()))

    
if __name__ == '__main__':
    main()

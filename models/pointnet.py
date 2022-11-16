# Architecture from: https://github.com/fxia22/pointnet.pytorch

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def feature_transform_regularizer(trans):
    d = trans.size(1)
    batchsize = trans.size(0)

    I = torch.eye(d, requires_grad=True).repeat(batchsize, 1, 1)

    if trans.is_cuda:
        I = I.cuda()

    difference = torch.bmm(trans, trans.transpose(2, 1)) - I
    loss = torch.mean(torch.norm(difference, dim=(1, 2)))

    return loss


class TNet(nn.Module):
    def __init__(self, pointDim = 3):
        super(TNet, self).__init__()

        self.pointDim = pointDim

        self.conv1 = nn.Sequential(nn.Conv1d(self.pointDim, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU())                                     
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU())                              
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.ReLU())

        self.pool = nn.Sequential(nn.AdaptiveMaxPool1d(1),
                                  nn.Flatten(1))

        self.fc1 = nn.Sequential(nn.Linear(1024, 512, bias=False),
                                 nn.BatchNorm1d(512),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(512, 256, bias=False),
                                 nn.BatchNorm1d(256),
                                 nn.ReLU())
        self.fc3 = nn.Linear(256, self.pointDim*self.pointDim)

    def forward(self, x):
        batchsize = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.pool(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        iden = torch.eye(self.pointDim, requires_grad=True).repeat(batchsize,1,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x.view(-1, self.pointDim, self.pointDim) + iden
        return x


class PointNetfeat(nn.Module):
    def __init__(self, emb_dims=1024, feature_transform=False):
        super(PointNetfeat, self).__init__()

        self.emb_dims = emb_dims
        self.feature_transform = feature_transform

        self.stn = TNet(pointDim=3)
        if self.feature_transform:
            self.fstn = TNet(pointDim=64)


        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU())                            
        self.conv2 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU())                                 
        self.conv3 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU())                                  
        self.conv4 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU())                              
        self.conv5 = nn.Sequential(nn.Conv1d(128, self.emb_dims, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(self.emb_dims),
                                   nn.ReLU())


        self.pool = nn.Sequential(nn.AdaptiveMaxPool1d(1),
                                  nn.Flatten(1))


    def forward(self, x):
        batch_size = x.size(0)

        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)

        x = self.conv1(x)
        x = self.conv2(x)

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)


        return x, trans, trans_feat


class PointNet(nn.Module):
    def __init__(self, numClass, emb_dims=1024, dropout_rate=0.5, feature_transform=False):
        super(PointNet, self).__init__()

        # Arguments:
        self.dropout_rate = dropout_rate
        self.feature_transform = feature_transform
        self.emb_dims = emb_dims

        self.feat = PointNetfeat(emb_dims = self.emb_dims, feature_transform=self.feature_transform)

        self.fc1 = nn.Sequential(nn.Linear(self.emb_dims, 512, bias=False),
                                 nn.BatchNorm1d(512),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(512, 256, bias=False),
                                 nn.BatchNorm1d(256),
                                 nn.ReLU())
        self.fc3 = nn.Linear(256, numClass)
        self.dp1 = nn.Dropout(p=self.dropout_rate)
        self.dp2 = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = self.fc1(x)
        x = self.dp1(x)
        x = self.fc2(x)
        x = self.dp2(x)
        x = self.fc3(x)
        return x, trans, trans_feat


if __name__ == '__main__':
    from torch.autograd import Variable

    sim_data = Variable(torch.rand(32, 3, 1024))
    trans = TNet(pointDim=3)
    out = trans(sim_data)
    print('TNet - 3', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 1024))
    trans = TNet(pointDim=64)
    out = trans(sim_data_64d)
    print('TNet - 64', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat()
    out, _, _ = pointfeat(sim_data)
    print('Feat', out.size())

    cls = PointNet(numClass=5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

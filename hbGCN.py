import torch
import torch_geometric as tg
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter,scatter_add
from torch.nn import Linear as Lin
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class associationP(torch.nn.Module):
    def __init__(self, input_dim, dropout=0.2):
        super(associationP, self).__init__()
        hidden=128
        self.Mlp =nn.Sequential(
                nn.Linear(input_dim, hidden*4, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden*4),
                nn.Linear(hidden*4, hidden*2, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden*2),
                nn.Dropout(dropout),
                nn.Linear(hidden*2, hidden, bias=True),
                )
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.input_dim = input_dim
        self.model_init()

    def forward(self, x):
        x1 = x[:,0:self.input_dim]
        x2 = x[:,self.input_dim:]
        h1 = self.Mlp(x1)
        h2 = self.Mlp(x2)
        p = (self.cos(h1,h2) + 1)*0.5
        return p

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

#DBwGcn: dynamic edge weights + Bayesian GCN
class DBwGcn(torch.nn.Module):
    def __init__(self, input_dim, num_classes, dropout, edgenet_input_dim, edge_dropout, hgc, lg,device='cuda'):
        super(DBwGcn, self).__init__()
        K=4       
        hidden = [hgc for i in range(lg)] 
        self.dropout = dropout
        self.edge_dropout = edge_dropout 
        bias = False 
        self.relu = torch.nn.ReLU(inplace=True) 
        self.lg = lg 
        self.gconv = nn.ModuleList()
        for i in range(lg):
            in_channels = input_dim if i==0  else hidden[i-1]
            self.gconv.append(tg.nn.ChebConv(in_channels, hidden[i], K, normalization='sym', bias=bias)) 
        cls_input_dim = sum(hidden) 

        self.Mlp2 = nn.Sequential(
                torch.nn.Linear(cls_input_dim, 256),
                torch.nn.ReLU(inplace=True),
                nn.BatchNorm1d(256), 
                torch.nn.Linear(256, num_classes))

        self.edge_net = associationP(input_dim=edgenet_input_dim//2, dropout=dropout)
        self.model_init()
        self.softmax=nn.Softmax(dim=1)
        self.input_dim = input_dim
        self.hidden_dim = 16
        self.output_dim = num_classes
        self.num_samples = 5
        self.device = device
        self.alpha = 1
        self.beta = 1
        self.prior_mean = torch.zeros(self.hidden_dim)
        self.prior_var = torch.ones(self.hidden_dim)
        self._layer = nn.Linear(input_dim, self.hidden_dim)
        self.classifier = nn.Linear(self.hidden_dim, self.output_dim)
        self.sigmoid = nn.Sigmoid()

    def Linear2(self, dimIn):
        xL = nn.Linear(dimIn, self.output_dim, bias=True)
        return xL.to(self.device)


    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def de_weights(self,edge_index, edgenet_input, enforce_edropout):
        if self.edge_dropout > 0:
            if enforce_edropout or self.training:
                one_mask = torch.ones([edgenet_input.shape[0], 1]).cuda()
                self.drop_mask = F.dropout(one_mask, self.edge_dropout, True)
                self.bool_mask = torch.squeeze(self.drop_mask.type(torch.bool))
                edge_index = edge_index[:, self.bool_mask]
                edgenet_input = edgenet_input[self.bool_mask]

        edge_weight = torch.squeeze(self.edge_net(edgenet_input))

        return edge_weight,edge_index

    def dense_gcn(self,features, edge_index, edge_weight):
        features = F.dropout(features, self.dropout, self.training)
        h = self.relu(self.gconv[0](features, edge_index, edge_weight))
        h0 = h

        for i in range(1, self.lg):
            h = F.dropout(h, self.dropout, self.training)
            h = self.relu(self.gconv[i](h, edge_index, edge_weight))
            jk = torch.cat((h0, h), axis=1)
            h0 = jk
        logit = self.Mlp2(jk)

        return logit

    def sample_posterior(self, h, edge_index, edge_weight):
        prior_h = torch.distributions.Normal(self.prior_mean, self.prior_var.sqrt())

        h_samples = []
        for _ in range(self.num_samples):
            perm = torch.randperm(h.size(0))
            chen2 = h[edge_index[0]] * edge_weight.unsqueeze(-1)
            agg_embed = scatter_add(chen2, edge_index[1], dim=0)
            for i in perm:
                ph = prior_h.sample()
                ph = ph.to(self.device)
                h[i] = ph + agg_embed[i]

            h_samples.append(h.clone())

        return torch.stack(h_samples)

    def bys_gcn(self,features, edge_index, edge_weight):
        x,edge_index,edge_weight=features, edge_index, edge_weight
        h = self._layer(x)

        h_samples = []
        for _ in range(self.num_samples):
            h_sample = self.sample_posterior(h, edge_index, edge_weight)  #(10,871,16)
            h_samples.append(h_sample)
        h_samples = torch.stack(h_samples) #(10,10,871,16)

        h_samples_2 = h_samples.view(x.shape[0], -1)

        hL = self.Linear2(h_samples_2.shape[1])  # (10,2)
        pred_logits = hL(h_samples_2.float())
        pred_prob = self.sigmoid(pred_logits)

        return pred_prob


    def forward(self, features, edge_index, edgenet_input, enforce_edropout=False):
        # de_weights:dynamic edge weights
        edge_weight,edge_index=self.de_weights(edge_index, edgenet_input, enforce_edropout)

        # dense_gcn:dense gcn   bys_gcn:bayesian gcn
        logit_1=self.dense_gcn(features, edge_index, edge_weight)
        logit_2 = self.bys_gcn(features, edge_index, edge_weight)
        logit_1_2 = self.softmax(logit_1)
        logit_2_2 = self.softmax(logit_2)
        pred_logits = torch.max(logit_1_2, logit_2_2)

        return logit_1_2,logit_2_2,pred_logits


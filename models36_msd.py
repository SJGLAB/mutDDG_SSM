import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv
import numpy as np

class GeometricEncoder(torch.nn.Module):
    def __init__(self, hidden_size, nheads=8):
        super(GeometricEncoder, self).__init__()
        num_atom_feature = 36
        nf = hidden_size
        n_out = hidden_size
        GAT = GATConv
        self.conv1 = GAT(num_atom_feature, int(nf / nheads), heads=nheads)
        self.conv2 = GAT(nf, int(nf / nheads), heads=nheads)
        self.conv3 = GAT(nf, int(nf / nheads), heads=nheads)
        self.conv4 = GAT(nf, int(n_out / nheads), heads=nheads)

        self.lin5 = torch.nn.Linear(256*2, 256)
        self.lelu = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(p=0.2)
        self.lin6 = torch.nn.Linear(256, 128)
        self.lin7 = torch.nn.Linear(128, 1)

    ################################################
    def regre(self, data):
        X = data.x
        E = data.edge_sparse
        Ea = data.edge_attr_sp
        rep_1, rep_2 = self.step(X, E, Ea)
        id_contact = torch.nonzero((X[:, 24] == 1).float()).view(-1)

        maxx41 = rep_1[id_contact, :]
        maxx42 = rep_2[id_contact, :]
        print(maxx41.shape)
        rep = torch.cat((maxx41, maxx42), 1)
        print(rep.shape)
        rep2 = self.dropout(self.lelu(self.lin5(rep)))
        rep3 = self.dropout(self.lelu(self.lin6(rep2)))
        rep4 = self.lin7(rep3)
        return rep4


    def step(self, X, E, Ea):
        edge_index, edge_attr = E.t(), Ea.view(-1)

        x_1 = self.conv1(X, edge_index)
        x_2 = self.conv2(x_1,edge_index)
        x_3 = self.conv3(x_2,edge_index)
        x_4 = self.conv4(x_3,edge_index)

        return x_3, x_4

    def gen_features(self, X, E, Ea, X_m, E_m, Ea_m):
        x3, x4 = self.step(X, E, Ea)
        x3_m, x4_m = self.step(X_m, E_m, Ea_m)

        idx = torch.nonzero((X[:, 24] == 1).float()).view(-1)
        idxm = torch.nonzero((X_m[:, 24] == 1).float()).view(-1)
        maxx4 = torch.max(x4[idx, :], 0)[0]
        maxx4m = torch.max(x4_m[idxm, :], 0)[0]
        meanx4 = torch.mean(x4[idx, :], 0)
        meanx4m = torch.mean(x4_m[idxm, :], 0)
        #
        maxx3 = torch.max(x3[idx, :], 0)[0]
        maxx3m = torch.max(x3_m[idxm, :], 0)[0]
        meanx3 = torch.mean(x3[idx, :], 0)
        meanx3m = torch.mean(x3_m[idxm, :], 0)
        #

        id_contact = torch.nonzero((X[:, 33] == 1).float()).view(-1)
        idm_contact = torch.nonzero((X_m[:, 33] == 1).float()).view(-1)
        # print(3)
        maxx4_contact = torch.max(x4[id_contact, :], 0)[0]
        maxx4m_contact = torch.max(x4_m[idm_contact, :], 0)[0]
        meanx4_contact = torch.mean(x4[id_contact, :], 0)
        meanx4m_contact = torch.mean(x4_m[idm_contact, :], 0)

        maxx3_contact = torch.max(x3[id_contact, :], 0)[0]
        maxx3m_contact = torch.max(x3_m[idm_contact, :], 0)[0]
        meanx3_contact = torch.mean(x3[id_contact, :], 0)
        meanx3m_contact = torch.mean(x3_m[idm_contact, :], 0)


        rep1 = [maxx4, maxx4m, meanx4, meanx4m, maxx4_contact, maxx4m_contact, meanx4_contact, meanx4m_contact,
                maxx3, maxx3m, meanx3, meanx3m, maxx3_contact, maxx3m_contact, meanx3_contact, meanx3m_contact, ]

        rep = torch.cat(rep1 + [maxx4 - maxx4m, meanx4 - meanx4m, maxx3 - maxx3m, meanx3 - meanx3m] +
                        [maxx4_contact - maxx4m_contact, meanx4_contact - meanx4m_contact,
                         maxx3_contact - maxx3m_contact, meanx3_contact - meanx3m_contact])

        # print(rep.shape)
        return rep



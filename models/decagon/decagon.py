import torch
import torch.nn.functional as F
import numpy as np
import params

from torch_scatter.scatter import scatter_add
from torch.nn import LSTM, Parameter, LeakyReLU
from torch_geometric.nn import SAGEConv

class Decagon(torch.nn.Module):
    def __init__(self, featureSize, embeddingSize, nSe, nD, nPro, nLayer=params.N_LAYER, device=torch.device('cpu')):
        super(Decagon, self).__init__()
        self.nSe = nSe
        self.nD = nD
        self.nV = nSe + nD
        self.embeddingSize = embeddingSize
        self.halfEmbeddingSize = embeddingSize // 2
        self.device = device
        self.feature2EmbedLayer1 = torch.nn.Linear(featureSize, embeddingSize)
        self.feature2EmbedLayer2 = torch.nn.Linear(embeddingSize, embeddingSize)
        self.embeddingSe = torch.nn.Embedding(nSe, embeddingSize)
        self.embeddingSe.weight.data.uniform_(0.001, 0.3)
        self.nLayer = nLayer

        self.feature2EmbedLayer1 = torch.nn.Linear(featureSize, embeddingSize).to(device)
        self.feature2EmbedLayer2 = torch.nn.Linear(embeddingSize, embeddingSize).to(device)

        self.proteinEmbedding = torch.nn.Embedding(nPro, embedding_dim=embeddingSize).to(device)
        self.pIds = torch.from_numpy(np.arange(0, nPro, 1)).long().to(device)
        # self.nodesEmbedding = torch.nn.Embedding(num_embeddings=numNode + 1, embedding_dim=params.EMBEDDING_SIZE)
        # self.nodesEmbedding.weight.data.uniform_(0.001, 0.3)
        # Molecule graph neural net

        self.convLayers = torch.nn.ModuleList()
        self.nLayer = nLayer

        for i in range(nLayer):
            layer = SAGEConv(params.EMBEDDING_SIZE, params.EMBEDDING_SIZE).to(device)
            self.convLayers.append(layer)

        self.outLayer1 = torch.nn.Linear(params.EMBEDDING_SIZE * 2, params.EMBEDDING_SIZE).to(device)
        self.outLayer2 = torch.nn.Linear(params.EMBEDDING_SIZE, nSe).to(device)
        # self.defaultAct = torch.nn.Hardshrink(lambd=0.00001)
        self.defaultAct = torch.nn.ReLU()
        self.dim1s = [i for i in range(self.nD)]
        self.dim2s = [i for i in range(self.nD)]
        self.dim3s = [i for i in range(self.nSe)]

        seIndices = [i for i in range(self.nSe)]
        self.seIndices = torch.from_numpy(np.asarray(seIndices)).long().to(self.device)

        self.linkAct = torch.nn.Hardshrink(lambd=0.000001)

    def getWLoss(self, target, pred, w=params.L_W):
        s = target.shape

        arx = torch.full(s, w).to(self.device)
        arx[target == 1] = 1
        e = target - pred
        e = e ** 2
        e = arx * e
        return torch.mean(e)

    def sampleDims(self, nsample=-1, isFull=False, toTuple=True):
        tp = None
        if isFull:
            d1, d2, dse = torch.from_numpy(np.arange(0, self.nD)).long().to(self.device), \
                          torch.from_numpy(np.arange(0, self.nD)).long().to(self.device), \
                          torch.from_numpy(np.arange(0, self.nSe)).long().to(self.device)
        else:
            d1, d2, dse = torch.from_numpy(np.random.choice(self.dim1s, nsample, replace=False)).long().to(self.device), \
                          torch.from_numpy(np.random.choice(self.dim2s, nsample, replace=False)).long().to(self.device), \
                          torch.from_numpy(np.random.choice(self.dim3s, nsample, replace=False)).long().to(self.device)

        if toTuple:
            td2 = d2.expand(nsample, -1).t().reshape(-1).expand(nsample, -1).reshape(-1)
            tdse = dse.expand(nsample, -1).reshape(-1).expand(nsample, -1).reshape(-1) + self.nD
            td1 = d1.expand(nsample * nsample, nsample).t().reshape(-1)
            tp = torch.vstack((td1, td2, tdse)).t().to(self.device)
        return d1, d2, dse, tp
    def sampleDims2(self, nsample=-1, isFull=False, toTuple=True):
        tp = None
        if isFull:
            d1, d2, dse = torch.from_numpy(np.arange(0, self.nD)).long().to(self.device), \
                          torch.from_numpy(np.arange(0, self.nD)).long().to(self.device), \
                          torch.from_numpy(np.arange(0, self.nSe)).long().to(self.device)
        else:
            d1, d2, dse = torch.from_numpy(np.random.choice(self.dim1s, nsample, replace=False)).long().to(self.device), \
                          torch.from_numpy(np.random.choice(self.dim2s, nsample, replace=False)).long().to(self.device), \
                          torch.from_numpy(np.random.choice(self.dim3s, nsample, replace=False)).long().to(self.device)

        if toTuple:
            td2 = d2.expand(nsample, -1).t().reshape(-1)
            # tdse = dse.expand(nsample, -1).reshape(-1).expand(nsample, -1).reshape(-1) + self.nD
            td1 = d1.expand(nsample, -1).reshape(-1)
            tp = torch.vstack((td1, td2)).t().to(self.device)
        return d1, d2, dse, tp
    def forward1(self, edge_index, drugFeatures):
        # print(self.feature2EmbedLayer1.weight.data[0, :])

        drugF = F.relu(self.feature2EmbedLayer1(drugFeatures))
        drugF = F.relu(self.feature2EmbedLayer2(drugF))

        proteinEmbedding = self.proteinEmbedding(self.pIds)
        xF = torch.cat((drugF, proteinEmbedding), dim=0)
        for iLayer, layerWeight in enumerate(self.convLayers):
            x = layerWeight(xF, edge_index)
            x = self.defaultAct(x)
        # Last layer:

        self.finalX = x[:self.nD]
        # print(self.finalX.shape)
        return self.finalX


    def forward2(self, x, tpl, sampleSes):
        # print(self.feature2EmbedLayer1.weight.data[0, :])

        xd1 = x[tpl[:, 0]]
        xd2 = x[tpl[:, 1]]
        x = torch.cat((xd1, xd2), dim=1).to(self.device)

        out1 = self.linkAct(self.outLayer1(x))
        out2 = self.linkAct(self.outLayer2(out1))
        out2 = out2.reshape(-1, self.nSe)[:, sampleSes]
        return out2.reshape(-1)

    def forward3(self, x, tpl, anchorLabels):
        # print(self.feature2EmbedLayer1.weight.data[0, :])

        xd1 = x[tpl[:, 0]]
        xd2 = x[tpl[:, 1]]
        x = torch.cat((xd1, xd2), dim=1).to(self.device)
        out1 = self.linkAct(self.outLayer1(x))
        out2 = self.linkAct(self.outLayer2(out1))
        out2 = out2.reshape(-1)
        return out2[anchorLabels]
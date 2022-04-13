import torch
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, GatedGraphConv, GATConv
from torch import sigmoid
import params

from torch.nn import Linear

from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

import math
import numpy as np


class Decagon(torch.nn.Module):
    def __init__(self, featureSize, embeddingSize, nSe, nD, nPro, nLayer=params.N_LAYER, device=torch.device('cpu'),
                 numNode=1000, layerType=SAGEConv):
        super(Decagon, self).__init__()

        self.nSe = nSe
        self.nD = nD
        self.nV = nSe + nD
        self.embeddingSize = embeddingSize
        self.device = device
        self.feature2EmbedLayer1 = torch.nn.Linear(featureSize, embeddingSize).to(device)
        self.feature2EmbedLayer2 = torch.nn.Linear(embeddingSize, embeddingSize).to(device)

        self.proteinEmbedding = torch.nn.Embedding(nPro, embedding_dim=embeddingSize).to(device)
        self.pIds = torch.from_numpy(np.arange(0, nPro, 1)).long().to(device)
        # self.nodesEmbedding = torch.nn.Embedding(num_embeddings=numNode + 1, embedding_dim=params.EMBEDDING_SIZE)
        # self.nodesEmbedding.weight.data.uniform_(0.001, 0.3)
        self.CONV_LAYER = layerType
        # Molecule graph neural net

        self.layerWeightList = []
        self.nLayer = nLayer
        for i in range(nLayer):
            layer = SAGEConv(params.EMBEDDING_SIZE, params.EMBEDDING_SIZE).to(device)
            self.layerWeightList.append(layer)
            self.register_parameter("sage%s" % i, layer.weight)

        self.outLayer1 = torch.nn.Linear(params.EMBEDDING_SIZE * 2, params.EMBEDDING_SIZE).to(device)
        self.outLayer2 = torch.nn.Linear(params.EMBEDDING_SIZE, nSe).to(device)

    def forward1(self, edge_index, drugFeatures):
        # print(self.feature2EmbedLayer1.weight.data[0, :])

        drugF = F.relu(self.feature2EmbedLayer1(drugFeatures))
        drugF = F.relu(self.feature2EmbedLayer2(drugF))

        proteinEmbedding = self.proteinEmbedding(self.pIds)
        xF = torch.cat((drugF, proteinEmbedding), dim=0)
        for iLayer, layerWeight in enumerate(self.layerWeightList):
            x = layerWeight(xF, edge_index)
            x = F.relu(x)
        # Last layer:

        self.finalX = x[:self.nD]
        # print(self.finalX.shape)
        return self.finalX

    def forward2(self, x, tpl, anchorLabels):
        # print(self.feature2EmbedLayer1.weight.data[0, :])

        xd1 = x[tpl[:, 0]]
        xd2 = x[tpl[:, 1]]
        x = torch.cat((xd1, xd2), dim=1).to(self.device)
        out1 = F.relu(self.outLayer1(x))
        out2 = F.relu(self.outLayer2(out1))
        out2 = out2.reshape(-1)
        return out2[anchorLabels]

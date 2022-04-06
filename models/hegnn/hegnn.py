import torch
import torch.nn.functional as F
import numpy as np
import params

from torch_scatter.scatter import scatter_add
from torch.nn import LSTM, Parameter, LeakyReLU


class HEGNN(torch.nn.Module):
    def __init__(self, featureSize, embeddingSize, nSe, nD, nLayer=params.N_LAYER, device=torch.device('cpu')):
        super(HEGNN, self).__init__()
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
        self.layerWeightList = []
        self.dimWeightList = []
        self.nLayer = nLayer

        self.drugBiLSTM = LSTM(input_size=embeddingSize, hidden_size=embeddingSize, proj_size=self.halfEmbeddingSize,
                               batch_first=True,
                               bidirectional=True).to(self.device)
        self.seBiLSTM = LSTM(input_size=embeddingSize, hidden_size=embeddingSize, proj_size=self.halfEmbeddingSize,
                             batch_first=True,
                             bidirectional=True).to(self.device)
        w = torch.zeros(2 * self.embeddingSize).uniform_(0.001, 0.1)
        self.w = Parameter(w, requires_grad=True).to(self.device)
        self.dim1s = [i for i in range(self.nD)]
        self.dim2s = [i for i in range(self.nD)]
        self.dim3s = [i for i in range(self.nSe)]

        seIndices = [i for i in range(self.nSe)]
        self.seIndices = torch.from_numpy(np.asarray(seIndices)).long().to(self.device)

        self.linkingPrediction1 = torch.nn.Linear(3 * embeddingSize, embeddingSize).to(self.device)
        self.linkingPrediction2 = torch.nn.Linear(embeddingSize, 1).to(self.device)
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

    def projectNonNegW(self):
        for dimWeight in self.dimWeightList:
            dimWeight.weight.data[dimWeight.weight.data < 0] = 0
            # dimWeight.weight.data[dimWeight.weight.data > 1] = 1
        self.lastDimWeight.weight.data[self.lastDimWeight.weight.data < 0] = 0

    def forward1(self, drugFeatures, sampledNodeList):
        # nD = drugFeatures.shape[0]
        # nSE = len(sampledSizeList) - nD

        drugF = F.relu(self.feature2EmbedLayer1(drugFeatures))
        drugF = F.relu(self.feature2EmbedLayer2(drugF))

        xSe = self.embeddingSe(self.seIndices)

        x = torch.cat((drugF, xSe), 0)

        inputDrugNeighbors = x[sampledNodeList[0]]
        inputSeNeghbors = x[sampledNodeList[1]]

        lstmDrugNeighbors, _ = self.drugBiLSTM(inputDrugNeighbors)
        lstmSeNeighbors, _ = self.seBiLSTM(inputSeNeghbors)

        lstmReses = [lstmDrugNeighbors, lstmSeNeighbors]
        lstms = [x]
        for i in range(2):
            lstm = torch.mean(lstmReses[i], dim=1)
            lstms.append(lstm)
        ss = 0
        wxvs = []
        for v in lstms:
            # assert not v.isnan().any()
            # assert not v.isinf().any()
            # assert not x.isnan().any()
            # assert not x.isinf().any()
            wxv = torch.exp(torch.matmul(torch.cat([x, v], dim=1), self.w))
            # assert not wxv.isnan().any()
            # assert not wxv.isinf().any()

            ss += wxv
            wxvs.append(wxv)

        # assert not ss.isnan().any()
        # assert not ss.isinf().any()

        ws2 = []
        for wxv in wxvs:
            vv = wxv / (ss + 1e-10)
            # print(1, wxv.shape, ss.shape)
            # print(2, torch.max(ss),  torch.min(ss), torch.max(wxv))
            # print(3, torch.max(vv), torch.min(vv))
            # assert not wxv.isnan().any()
            # assert not ss.isnan().any()
            # assert not vv.isnan().any()

            ws2.append(vv)
        finalX = 0
        for i in range(3):
            fx = lstms[i]
            w = ws2[i]
            # assert not w.isnan().any()
            # print("SS" , fx.shape, w.shape)
            finalX += torch.mul(w.unsqueeze(-1), fx)
        assert not finalX.isnan().any()
        # print(finalX.shape)
        self.finalX = finalX

        return self.finalX

    def forward2(self, x, tpl):

        xd1 = x[tpl[:, 0]]
        xd2 = x[tpl[:, 1]]
        xse = x[tpl[:, 2]]

        x = torch.cat([xd1, xd2, xse], dim=1)
        o1 = self.linkAct(self.linkingPrediction2(self.linkAct(self.linkingPrediction1(x))))
        return o1

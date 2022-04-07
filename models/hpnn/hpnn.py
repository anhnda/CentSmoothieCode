import torch
import torch.nn.functional as F
import numpy as np
import params


class HPNN(torch.nn.Module):
    def __init__(self, featureSize, embeddingSize, nSe, nD, nLayer=params.N_LAYER, device=torch.device('cpu')):
        super(HPNN, self).__init__()
        self.nSe = nSe
        self.nLayer = nLayer
        self.device = device
        self.nD = nD
        self.feature2EmbedLayer1 = torch.nn.Linear(featureSize, embeddingSize).to(self.device)
        self.feature2EmbedLayer2 = torch.nn.Linear(embeddingSize, embeddingSize).to(self.device)
        self.embeddingSe = torch.nn.Embedding(nSe, embeddingSize).to(self.device)
        self.embeddingSe.weight.data.uniform_(0.001, 0.3)

        self.layerWeightList = []

        for i in range(nLayer):
            layer = torch.nn.Linear(embeddingSize, embeddingSize, bias=True).to(self.device)
            self.register_parameter("layerWeight" + "%s" % i, layer.weight)
            self.register_parameter("layerWBias" + "%s" % i, layer.bias)
            self.layerWeightList.append(layer)

        self.dim1s = [i for i in range(self.nD)]
        self.dim2s = [i for i in range(self.nD)]
        self.dim3s = [i for i in range(self.nSe)]

        seIndices = [i for i in range(self.nSe)]
        self.seIndices = torch.from_numpy(np.asarray(seIndices)).long().to(self.device)

        self.linkingPrediction1 = torch.nn.Linear(3 * embeddingSize, embeddingSize).to(self.device)
        self.linkingPrediction2 = torch.nn.Linear(embeddingSize, 1).to(self.device)
        self.linkAct = torch.nn.Hardshrink(lambd=0.000001)
        self.defaultAct = torch.nn.Hardshrink(lambd=0.000001)

    def forward1(self,  drugFeatures, A):
        # print(self.feature2EmbedLayer1.weight.data[0, :])

        drugF = F.relu(self.feature2EmbedLayer1(drugFeatures))
        drugF = F.relu(self.feature2EmbedLayer2(drugF))

        seIndices = [i for i in range(self.nSe)]
        indices = torch.from_numpy(np.asarray(seIndices)).long().to(self.device)
        xSe = self.embeddingSe(indices)
        x = torch.cat((drugF, xSe), 0)

        for iLayer, layerWeight in enumerate(self.layerWeightList):
            x = torch.matmul(A, x)
            if params.LAYER_WEIGHT:
                x = layerWeight(x)
            x = self.defaultAct(x)

        self.finalX = x

        return self.finalX


    def forward2(self, x, tpl, rsqrtdeg = None):

        xd1 = x[tpl[:, 0]]
        xd2 = x[tpl[:, 1]]
        xse = x[tpl[:, 2]]
        if rsqrtdeg is not None:
            dd1 = rsqrtdeg[tpl[:, 0]][:, None]
            dd2 = rsqrtdeg[tpl[:, 1]][:, None]
            dd3 = rsqrtdeg[tpl[:, 2]][:, None]

            xd1 = xd1 * dd1
            xd2 = xd2 * dd2
            xse = xse * dd3
        x = torch.cat([xd1, xd2, xse], dim=1)
        o1 = self.linkAct(self.linkingPrediction2(self.linkAct(self.linkingPrediction1(x))))
        return o1


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
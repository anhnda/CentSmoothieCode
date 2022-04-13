import torch
import torch.nn.functional as F
import numpy as np
import params

from torch_scatter.scatter import scatter_add


class CentSmoothie(torch.nn.Module):
    def __init__(self, featureSize, embeddingSize, nSe, nD, nLayer=params.N_LAYER, device=torch.device('cpu')):
        super(CentSmoothie, self).__init__()
        self.nSe = nSe
        self.nD = nD
        self.nV = nSe + nD
        self.embeddingSize = embeddingSize
        self.device = device
        self.feature2EmbedLayer1 = torch.nn.Linear(featureSize, embeddingSize)
        self.feature2EmbedLayer2 = torch.nn.Linear(embeddingSize, embeddingSize)
        self.embeddingSe = torch.nn.Embedding(nSe, embeddingSize)
        self.embeddingSe.weight.data.uniform_(0.001, 0.3)
        self.layerWeightList = []
        self.dimWeightList = []
        self.nLayer = nLayer

        self.dim1s = [i for i in range(self.nD)]
        self.dim2s = [i for i in range(self.nD)]
        self.dim3s = [i for i in range(self.nSe)]
        for i in range(nLayer):
            layer = torch.nn.Linear(embeddingSize, embeddingSize, bias=True).to(self.device)
            self.register_parameter("layerWeight" + "%s" % i, layer.weight)
            self.register_parameter("layerWBias" + "%s" % i, layer.bias)
            self.layerWeightList.append(layer)

        for i in range(nLayer):
            dimWeight = torch.nn.Embedding(nSe, embeddingSize).to(self.device)

            dimWeight.share_memory()
            # dimWeight.weight.data.uniform_(0.001, 0.3)
            dimWeight.weight.data.fill_(1)

            if not params.LEARN_WEIGHT_IN:
                dimWeight.weight.requires_grad = False
                dimWeight.weight.data.fill_(1)
            self.dimWeightList.append(dimWeight)
            self.register_parameter("dimWeight" + "%s" % i, dimWeight.weight)

        # Last dimWeight:
        lastDimWeight = torch.nn.Embedding(nSe, embeddingSize).to(self.device)
        lastDimWeight.share_memory()
        lastDimWeight.weight.data.uniform_(0.001, 1)

        if not params.LEARN_WEIGHT_LAST:
            lastDimWeight.weight.requires_grad = False
            lastDimWeight.weight.data.fill_(1)

        self.lastDimWeight = lastDimWeight

        self.dOnes = torch.ones(self.nV).to(self.device)
        self.diagI = torch.diag(torch.ones(self.nV)).to(self.device)
        seIndices = [i for i in range(self.nSe)]
        self.seIndices = torch.from_numpy(np.asarray(seIndices)).long().to(self.device)
        self.defaultAct = torch.nn.Hardshrink(lambd=0.000001)

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

    def construct1L(self, iDim, pos, wids, weights, iLayer=0, export=False):

        if iLayer == -1:
            wi = self.lastDimWeight
        else:
            wi = self.dimWeightList[iLayer]

        ws = wi(wids)
        ws = ws[:, iDim]
        ws = ws * weights
        nSize = self.nV * self.nV
        x = scatter_add(ws, pos, dim_size=nSize)

        L2 = x.reshape(self.nV, self.nV)
        L2 = L2 + L2.t()
        diag = torch.diag(torch.diag(L2))
        L2 = L2 - diag / 2

        L = L2
        assert torch.sum(L) < 10

        return L

    def constructCentL(self, pos, wids, weights, iLayer=0, export=False):
        if iLayer == -1:
            wi = self.lastDimWeight
        else:
            wi = self.dimWeightList[iLayer]

        nDim = wi.weight.shape[1]
        assert nDim == self.embeddingSize
        sSize = self.nV * self.nV
        aSize = sSize * nDim

        xpos = pos.repeat(nDim)
        xweights = weights.repeat(nDim)
        sz = pos.shape[0]
        ws = wi(wids).t().reshape(-1)
        ws = ws * xweights
        for iDim in range(nDim):
            xpos[iDim * sz: (iDim + 1) * sz] += iDim * sSize
        x = scatter_add(ws, xpos, dim_size=aSize)
        LL = x.reshape((nDim, self.nV, self.nV))
        for iDim in range(nDim):
            Li = LL[iDim]
            L2 = Li + Li.t()
            diag = torch.diag(torch.diag(L2)).to(self.device)
            L2 = L2 - diag / 2
            assert torch.sum(L2) < 10
            LL[iDim] = L2
        return LL

    # def constructCentL(self, pos, cors, wids, wi, nV):
    #     # Parameters:
    #     # nV: Number of nodes
    #     # wi: Embedding for side effect weights
    #     # pos: Serialized positions of each side effect
    #     #      weight to the corresponding position (i,j)
    #     #      in the Laplacian matrix, computed by:
    #     #      i * nV + j
    #     # cors: Coefficients corresponding to pos
    #     # Return:
    #     # Central-smoothing hypergraph Laplacian matrices for
    #     # K dimensions.
    #
    #     K = wi.weight.shape[1]
    #     sSize = nV * nV
    #     aSize = sSize * K
    #     xpos = pos.repeat(K)
    #     xcors = cors.repeat(K)
    #     sz = pos.shape[0]
    #     ws = wi(wids).t().reshape(-1)
    #     ws = ws * xcors
    #     for iDim in range(K):
    #         xpos[iDim * sz: (iDim + 1) * sz] += iDim * sSize
    #     x = scatter_add(ws, xpos, dim_size=aSize)
    #     LL = x.reshape((K, nV, nV))
    #     for iDim in range(K):
    #         Li = LL[iDim]
    #         L2 = Li + Li.t()
    #         diag = torch.diag(torch.diag(L2)).to(self.device)
    #         L2 = L2 - diag / 2
    #         LL[iDim] = L2
    #     return LL

    def normDegL(self, L):
        d = torch.diag(L)
        diag = torch.diag(d)

        A = diag - L
        A2 = A + self.diagI

        D = torch.sum(A2, dim=1)
        D = torch.pow(torch.sqrt(D), -1)
        DM12 = torch.diag(D)
        normA = torch.matmul(torch.matmul(DM12, A2), DM12)

        assert torch.min(d) >= 0
        assert torch.min(D) > 0

        return normA, d

    def normDegL2(self, L):
        d = torch.diag(L)
        diag = torch.diag(d)
        A = diag - L
        A2 = A + self.diagI

        normA = A2 / torch.max(A2)

        return normA, self.dOnes

    def forward1(self, drugFeatures, dd):
        drugF = F.relu(self.feature2EmbedLayer1(drugFeatures))
        drugF = F.relu(self.feature2EmbedLayer2(drugF))

        xSe = self.embeddingSe(self.seIndices)
        x = torch.cat((drugF, xSe), 0)

        pos, wids, weights = dd
        self.finalD = []

        if not params.LEARN_WEIGHT_IN:
            Li = self.construct1L(0, pos, wids, weights, 0, export=True)
            if params.ON_REAL:
                A, D = self.normDegL(Li)
            else:
                A, _ = self.normDegL2(Li)

        for iLayer, layerWeight in enumerate(self.layerWeightList):
            if params.LEARN_WEIGHT_IN:
                lList = self.constructCentL(pos, wids, weights, iLayer)
                AA = torch.empty((self.embeddingSize, self.nV, self.nV)).to(self.device)
                for iDim in range(self.embeddingSize):
                    Li = lList[iDim].squeeze()
                    if params.ON_REAL:
                        Ai, _ = self.normDegL(Li)
                    else:
                        Ai, _ = self.normDegL2(Li)
                    AA[iDim] = Ai

                x = x.t().unsqueeze(-1)
                x2 = torch.bmm(AA, x)
                x2 = x2.squeeze().t()

                if params.LAYER_WEIGHT:
                    x2 = layerWeight(x2)
                x = self.defaultAct(x2)
            else:
                x = torch.matmul(A, x)
                if params.LAYER_WEIGHT:
                    x = layerWeight(x)
                x = self.defaultAct(x)

        x = F.relu(x)
        # Last layer:
        if params.ON_REAL:
            lList = self.constructCentL(pos, wids, weights, -1)
            for iDim in range(self.embeddingSize):
                Li = lList[iDim].squeeze()
                Ai, Di = self.normDegL(Li)
                # print("Last A D: ", Ai[0, :10], Di[:10])
                Di[Di == 0] = 1
                rsqrtD = torch.pow(torch.sqrt(Di), -1)
                self.finalD.append(rsqrtD)

        else:
            rsqrtD = self.dOnes
            for iDim in range(self.embeddingSize):
                self.finalD.append(rsqrtD)

        self.finalX = x

        return self.finalX

    def forward2(self, x, tpl):

        xd1 = x[tpl[:, 0]]
        xd2 = x[tpl[:, 1]]
        xse = x[tpl[:, 2]]
        v = 0
        seIds = tpl[:, 2] - self.nD
        w = self.lastDimWeight(seIds)
        for iDim in range(self.embeddingSize):
            xd1i = xd1[:, iDim]
            xd2i = xd2[:, iDim]
            xsei = xse[:, iDim]
            wei = w[:, iDim]

            rsqrtdeg = self.finalD[iDim]
            dd1 = rsqrtdeg[tpl[:, 0]]
            dd2 = rsqrtdeg[tpl[:, 1]]
            dd3 = rsqrtdeg[tpl[:, 2]]

            xd1i = xd1i * dd1
            xd2i = xd2i * dd2
            xsei = xsei * dd3
            vi = (xd1i + xd2i) / 2 - xsei
            vi = vi * wei
            vi = torch.squeeze(vi)
            vi = torch.mul(vi, vi)
            v += vi

        smt = v
        out = smt + 1 + 1e-2
        out2 = 1 / out
        return out2

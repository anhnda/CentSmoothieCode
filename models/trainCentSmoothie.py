from models.centsmoothie import CentSmoothie

import torch
import numpy as np
import inspect
import params
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from utils import utils
import time


def getMSE(a1, a2):
    v = a1 - a2
    v = np.multiply(v, v)
    return np.sqrt(np.sum(v) / (v.shape[0] * v.shape[0]))


class WrapperWeightCentSmooth:
    def __init__(self, device=torch.device('cpu')):

        self.device = torch.device(params.GPU_DEVICE if torch.cuda.is_available() else 'cpu')
        self.name = "CentSmoothie"
        self.isFitAndPredict = True

    def setLogger(self, logger):
        self.logger = logger
        self.logger.infoAll(inspect.getsource(CentSmoothie))

    def getLoss(self, out1, out2, reg=None):
        # print(torch.min(out1), torch.max(out1), torch.min(out2), torch.max(out2))
        e1 = - torch.sum(torch.log(out1)) - torch.sum(torch.log(1 - out2))
        if reg is not None:
            if params.R_TYPE == "L1":
                e1 += params.LAMBDA_R * torch.sum(torch.abs(reg))
            else:  # L2:
                e1 += params.LAMBDA_R * torch.sum(torch.mul(reg, reg))
        return e1

    def convertDictNp2LongTensor(self, d):
        d2 = dict()
        for k, v in d.items():
            v = torch.from_numpy(v).long()
            d2[k] = v
        return d2

    def list2Pair(self, l):
        dCount = dict()
        for v in l:
            utils.add_dict_counter(dCount, v)
        ks = []
        vs = []
        for k, v in dCount.items():
            ks.append(k)
            vs.append(v)

        ks = np.asarray(ks)
        vs = np.asarray(vs)
        return ks, vs

    def convertDictNp2PairLongTensor(self, d):
        d2 = dict()
        for k, v in d.items():
            ks, vs = self.list2Pair(v)
            ks = torch.from_numpy(ks).long().to(self.device)
            vs = torch.from_numpy(vs).float().to(self.device)
            d2[k] = (ks, vs)
        return d2

    def convertAllDict2LongList(self, dd, dimSize):
        pos = []
        wids = []
        weights = []
        for ix, d in enumerate(dd):
            if ix == 0 or ix == 1:
                w = 0.25
            elif ix == 2:
                w = -0.5
            else:
                w = 1
            for k, v in d.items():
                i, j = k
                assert i <= j
                ps = i * dimSize + j
                ks, vs = self.list2Pair(v)
                for jj in range(ks.shape[0]):
                    wid = ks[jj]
                    count = vs[jj]
                    wids.append(wid)
                    weights.append(count * w)
                    pos.append(ps)
        pos = torch.from_numpy(np.asarray(pos)).long().to(self.device)
        wids = torch.from_numpy(np.asarray(wids)).long().to(self.device)
        weights = torch.from_numpy(np.asarray(weights)).float().to(self.device)

        return pos, wids, weights

    def selectSubIndices(self, secs, tpls):
        secs = secs[::-1]
        sz = len(secs)
        indices = [[] for _ in range(sz)]
        for ii, tpl in enumerate(tpls):
            _, _, adrIdOf = tpl
            i = sz
            for i, sec in enumerate(secs):
                if adrIdOf in sec:
                    break
            for j in range(i, sz):
                indices[j].append(ii)

        res = []
        for indice in indices:
            res.append(np.asarray(indice))
        return res



    def train(self, dataWrapper, iFold, method="New", printDB=params.PRINT_DB):
        realData = dataWrapper.data
        target = dataWrapper.ddiTensorInDevice
        model = CentSmoothie(realData.featureSize, params.EMBEDDING_SIZE, realData.nSe, realData.nD, device=self.device)
        self.model = model.to(self.device)

        if params.OPTIMIZER == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        else:
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=0.01)

        trainTpl, testTpl, validTpl, negTestTpl, negValidTpl = realData.trainFold, realData.testFold, \
                                                               realData.validFold, realData.negFold, realData.negFold

        sortedADRs = realData.orderADRIds

        secs = [set() for _ in range(params.N_SEC)]
        secsList = [[] for _ in range(params.N_SEC)]
        secLength = int(len(sortedADRs) / params.N_SEC)
        for i, v in enumerate(sortedADRs):
            secId = int(i / secLength)
            if secId == params.N_SEC:
                secId = params.N_SEC - 1
            secs[secId].add(v + realData.nD)
            secsList[secId].append(v + realData.nD)
        adrSecIndiceTestPos = self.selectSubIndices(secs, testTpl)
        adrSecINdiceTestNeg = self.selectSubIndices(secs, negTestTpl)



        # A = realData.AFold
        # D = realData.DFold

        # print(A.shape, A[0, :10])
        # print(D.shape, D[:10])

        dd = self.convertAllDict2LongList(realData.trainPairStats, model.nV)

        # trainIds = torch.from_numpy(np.asarray(trainTpl)).long().to(self.device)
        testIds = torch.from_numpy(np.asarray(testTpl)).long().to(self.device)
        validIds = torch.from_numpy(np.asarray(validTpl)).long().to(self.device)

        negTestIds = torch.from_numpy(np.asarray(negTestTpl)).long().to(self.device)
        # negValidIds = torch.from_numpy(np.asarray(negValidTpl)).long()
        drugFeatures = torch.from_numpy(realData.drug2Features).float().to(self.device)

        arAUCAUPR = []
        arAUCVal = []

        arSecs = []

        startTime = time.time()

        finalX = None

        nd = min(params.N_SGD, model.nD, model.nSe)
        allResValues = []

        for i in range(params.N_ITER):
            optimizer.zero_grad()

            finalX = self.model.forward1(drugFeatures, dd)

            vd1, vd2, vd3, sampleTrain = self.model.sampleDims(nsample=nd, toTuple=True)
            out = self.model.forward2(finalX, sampleTrain)
            out = out.reshape(nd, nd, nd)

            targetx = target[vd1, :, :]
            targetx = targetx[:, vd2, :]
            targetx = targetx[:, :, vd3]

            errTrain = self.model.getWLoss(targetx, out)

            assert not torch.isnan(errTrain).any()

            errTrain.backward()
            optimizer.step()
            self.model.projectNonNegW()

            if i % params.ITER_DB == 0:
                print("\r@Iter ", i, end=" ")
                with torch.no_grad():
                    print(torch.max(self.model.dimWeightList[0].weight), torch.min(self.model.dimWeightList[0].weight))
                    print(torch.max(self.model.lastDimWeight.weight), torch.min(self.model.lastDimWeight.weight))

                    outTest = self.model.forward2(finalX, testIds).cpu().detach().numpy()
                    outValid = self.model.forward2(finalX, validIds).cpu().detach().numpy()
                    outNegTest = self.model.forward2(finalX, negTestIds).cpu().detach().numpy()

                    print(outTest[:20])
                    print(outValid[:20])
                    print(outNegTest[:20])

                    if params.ON_REAL:
                        reSec = []
                        for kk in range(params.N_SEC):
                            indicePos = adrSecIndiceTestPos[kk]
                            indiceNeg = adrSecINdiceTestNeg[kk]
                            outPosK = outTest[indicePos]
                            outNegK = outNegTest[indiceNeg]
                            auck, auprk = evalAUCAUPR1(outPosK, outNegK)
                            reSec.append([auck, auprk])
                            if (kk == params.N_SEC - 1):
                                allResValues.append([outPosK, outNegK])
                        arSecs.append(reSec)

                    auc, aupr = evalAUCAUPR1(outTest, outNegTest)
                    arAUCAUPR.append((auc, aupr))
                    aucv, auprv = evalAUCAUPR1(outValid, outNegTest)
                    arAUCVal.append(aucv)

                    cTime = time.time()
                    self.logger.infoAll((auc, aucv, aupr, "Elapse@:", i, cTime - startTime))

        selectedInd = np.argmax(arAUCVal)
        auc, aupr = arAUCAUPR[selectedInd]
        vv = -1
        if params.ON_REAL:
            vv = arSecs[selectedInd]
            print(vv)
            np.savetxt("%s%sW_%s" % (params.EMBEDDING_PREX, method, iFold), finalX.cpu().detach().numpy())
            np.savetxt("%s%sW_Weight%s" % (params.EMBEDDING_PREX, method, iFold),
                       self.model.lastDimWeight.weight.cpu().detach().numpy())
            predictedValues = allResValues[selectedInd]
            utils.save_obj(predictedValues, "%s/SaveCalValues_W_%s_%s" % (params.OUTPUT_DIR, method, iFold))

        return auc, aupr, vv


def evalAUCAUPR1(outPos, outNeg):
    s1 = outPos.shape[0]
    s2 = outNeg.shape[0]
    trueOut = np.zeros(s1 + s2)
    for i in range(s1):
        trueOut[i] = 1
    predicted = np.concatenate((outPos, outNeg))
    aupr = average_precision_score(trueOut, predicted)
    auc = roc_auc_score(trueOut, predicted)
    return auc, aupr


if __name__ == "__main__":
    pass

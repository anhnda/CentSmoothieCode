import random

from models.decagon.decagon import Decagon

import torch
import numpy as np
import inspect
import params
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from utils import utils
import time
from collections import Counter


def getMSE(a1, a2):
    v = a1 - a2
    v = np.multiply(v, v)
    return np.sqrt(np.sum(v) / (v.shape[0] * v.shape[0]))


class WrapperDecagon:
    def __init__(self, device=torch.device('cpu')):

        self.device = torch.device(params.GPU_DEVICE if torch.cuda.is_available() else 'cpu')
        self.name = "Decagon"
        self.isFitAndPredict = True

    def setLogger(self, logger):
        self.logger = logger
        self.logger.infoAll(inspect.getsource(Decagon))

    def getLoss(self, out1, out2, reg=None):
        # print(torch.min(out1), torch.max(out1), torch.min(out2), torch.max(out2))
        e1 = - torch.sum(torch.log(out1)) - torch.sum(torch.log(1 - out2))
        if reg is not None:
            if params.R_TYPE == "L1":
                e1 += params.LAMBDA_R * torch.sum(torch.abs(reg))
            else:  # L2:
                e1 += params.LAMBDA_R * torch.sum(torch.mul(reg, reg))
        return e1

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

    def visual2(self, trainTpl, seId, nD, finalX, method, selectedPairs=[], iFold=5, dId2SeName={}, dId2DrugName={}):
        finalX = finalX / np.max(np.fabs(finalX))
        print("MAX V", np.max(finalX))
        print(selectedPairs)
        drugIDSet = set()
        seIdOf = seId + nD
        drugPairSet = []
        for tpl in trainTpl:
            d1, d2, s = tpl
            if s == seIdOf:
                drugPairSet.append([d1, d2])
                drugIDSet.add(d1)
                drugIDSet.add(d2)

        mxPair = len(drugPairSet)
        drugIDList = list(drugIDSet)
        from postprocessing.visualization import plotData2

        title = r'$\mathrm{CentSmoothie}$'
        plotData2(finalX, "%s_%s" % (method, iFold), title, offset=nD, sid=seIdOf, dPairs=drugPairSet[:mxPair],
                  selectVDrugPair=selectedPairs, drugIDList=drugIDList, dSe2Name=dId2SeName,
                  dDrug2Name=dId2DrugName)

    def exportTopNeg(self, dSeId2Tpls, dSeId2Indices, NegRes, nD, dADR2Name, dDrug2Name, outFile):
        seOfIds = sorted(dSeId2Indices.keys())
        sorteddSeId2Tpls = dict()
        sortedSeId2Scores = dict()
        for seOfId in seOfIds:
            indices = dSeId2Indices[seOfId]
            tpls = dSeId2Tpls[seOfId]

            res = NegRes[indices]
            assert len(res) == len(tpls)

            sortedIndiceScores = np.argsort(res)[::-1]
            assert res[sortedIndiceScores[0]] >= res[sortedIndiceScores[1]]
            # print(res[sortedIndiceScores[0]])
            rr = []
            orr = dSeId2Tpls[seOfId]
            rscore = []
            for idx in sortedIndiceScores:
                d1, d2, _ = orr[idx]
                rr.append((d1, d2))
                rscore.append(res[idx])

            sorteddSeId2Tpls[seOfId - nD] = rr
            sortedSeId2Scores[seOfId - nD] = rscore
        fout = open(outFile, "w")
        for k, v in sorteddSeId2Tpls.items():
            adrName = dADR2Name[k]
            drugPairs = v
            rscore = sortedSeId2Scores[k]
            fout.write("%s\n" % adrName)
            for ii, pair in enumerate(drugPairs):
                d1, d2 = pair
                fout.write("\t%s, %s, %s\n" % (dDrug2Name[d1], dDrug2Name[d2], rscore[ii]))
            fout.write("\n_________\n")

        fout.close()

    def selectSubIndices2(self, secs, labelSegs, nD):
        secs = secs[::-1]
        sz = len(secs)
        indices = [[] for _ in range(sz)]

        for i, labelSeg in enumerate(labelSegs):
            adrIdOf = i + nD
            for j, sec in enumerate(secs):
                if adrIdOf in sec:
                    break
            for p in labelSeg:
                for kk in range(j, sz):
                    indices[kk].append(p)

        res = []
        for indice in indices:
            res.append(np.asarray(indice))
        return res

    def loadNegIds(self, ses, dSe2PairAll, dSe2IndicesAll):

        dSeId2Tpls = dict()

        dSeId2Indices = dict()

        for seId in ses:
            dSeId2Tpls[seId] = dSe2PairAll[seId]
            dSeId2Indices[seId] = dSe2IndicesAll[seId]

        return dSeId2Tpls, dSeId2Indices

    def convertPairLabel(self, ddPos, ddNeg, sz):
        pairs = []
        allAnchor = []
        trueLabels = []
        ind = 0
        posPairs = ddPos.keys()
        for pair, labels in ddPos.items():
            negLabels = utils.get_dict(ddNeg, pair, [])
            offset = ind * sz
            ind += 1
            pairs.append(pair)
            for l in labels:
                allAnchor.append(offset + l)
                trueLabels.append(1)
            for l in negLabels:
                allAnchor.append(offset + l)
                trueLabels.append(0)
        for pair, labels in ddNeg.items():
            if pair not in posPairs:
                offset = ind * sz
                ind += 1
                pairs.append(pair)
                for l in labels:
                    allAnchor.append(offset + l)
                    trueLabels.append(0)

        return torch.from_numpy(np.asarray(pairs)).long().to(self.device), torch.from_numpy(
            np.asarray(allAnchor)).long().to(self.device), \
               torch.from_numpy(np.asarray(trueLabels)).float().to(self.device)

    def convertPairLabelWithLabelP(self, ddPos, ddNeg, sz):
        pairs = []
        allAnchor = []
        trueLabels = []
        ind = 0
        posPairs = ddPos.keys()

        labelSegs = [[] for _ in range(sz)]
        dSe2NegPair = dict()
        dSe2NegIndices = dict()
        for i in range(sz):
            dSe2NegPair[i] = []
            dSe2NegIndices[i] = []

        for pair, labels in ddPos.items():
            negLabels = utils.get_dict(ddNeg, pair, [])
            offset = ind * sz
            ind += 1
            pairs.append(pair)
            for l in labels:
                allAnchor.append(offset + l)
                labelSegs[l].append(len(trueLabels))
                trueLabels.append(1)

            for l in negLabels:
                allAnchor.append(offset + l)
                indxx = len(trueLabels)
                labelSegs[l].append(indxx)
                dSe2NegPair[l].append(pair)
                dSe2NegIndices[l].append(indxx)
                trueLabels.append(0)

        for pair, labels in ddNeg.items():
            if pair not in posPairs:
                offset = ind * sz
                ind += 1
                pairs.append(pair)
                for l in labels:
                    allAnchor.append(offset + l)
                    indxx = len(trueLabels)
                    labelSegs[l].append(indxx)
                    dSe2NegPair[l].append(pair)
                    dSe2NegIndices[l].append(indxx)

                    trueLabels.append(0)

        return torch.from_numpy(np.asarray(pairs)).long().to(self.device), torch.from_numpy(
            np.asarray(allAnchor)).long().to(self.device), \
               torch.from_numpy(np.asarray(trueLabels)).float().to(self.device), labelSegs, dSe2NegPair, dSe2NegIndices

    def train(self, dataWrapper, iFold, method="New", printDB=params.PRINT_DB):
        realData = dataWrapper.data
        target = dataWrapper.ddiTensorInDevice
        model = Decagon(realData.featureSize, params.EMBEDDING_SIZE, nSe=realData.nSe, nD=realData.nD,
                        nPro=realData.nPro, nLayer=params.N_LAYER, device=self.device)

        self.model = model.to(self.device)

        if params.OPTIMIZER == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        else:
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=0.01)

        drugFeatures = torch.from_numpy(realData.drug2Features).float().to(self.device)
        edgeIndex = torch.tensor(realData.ppGraph, dtype=torch.long).t().contiguous().to(self.device)

        # edge2Label = realData.pTrainPair2Label
        validPosPair2Label = realData.pValidPosLabel
        testPosPair2Label = realData.pTestPosLabel
        testNegPair2Label = realData.pTestNegLabel

        # trainPairs, trainAnchor, trainLabel = self.convertPairLabel(edge2Label, testNegPair2Label, realData.nSe)
        # validPairs, validAnchor, validLabel = self.convertPairLabel(validPosPair2Label, testNegPair2Label, realData.nSe)
        testPairs, testAnchor, testLabel, labelSegs, dSe2NegPairAll, dSe2NegIndicesAll = self.convertPairLabelWithLabelP(
            testPosPair2Label, testNegPair2Label, realData.nSe)

        validPairs, validAnchor, validLabel, _, _, _ = self.convertPairLabelWithLabelP(
            validPosPair2Label, testNegPair2Label, realData.nSe)

        sortedADRs = realData.orderADRIds
        secs = [set() for _ in range(params.N_SEC)]
        secsList = [[] for _ in range(params.N_SEC)]

        secLength = int(len(sortedADRs) / params.N_SEC)
        for i, v in enumerate(sortedADRs):
            secId = int(i / secLength)
            if secId == params.N_SEC:
                secId = params.N_SEC - 1
            secs[secId].add(v + realData.nD)
            secsList[secId].append(v)

        segTests = self.selectSubIndices2(secs, labelSegs, realData.nD)

        if params.EXPORT_TOP_NEG:
            print("Exporting TOP NEGs...")
            dAdrName, dDrugName = utils.load_obj(params.ID2NamePath_TWOSIDE)
            outFile = "%s/Export_TOP_NEG_%s_%s" % (params.OUTPUT_DIR, method, iFold)

            predictedValues = utils.load_obj("%s/SaveCalValues_%s_%s" % (params.OUTPUT_DIR, "Decagon", iFold))
            _, outNegK = predictedValues
            ses = secsList[-1][-50:]
            dSeId2Tpls, dSeId2Indices = self.loadNegIds(ses, dSe2NegPairAll, dSe2NegIndicesAll)
            self.exportTopNeg(dSeId2Tpls, dSeId2Indices, outNegK, dAdrName, dDrugName, outFile)
            exit(-1)

        arAUCAUPR = []
        arAUCVal = []

        startTime = time.time()
        arSecs = []

        nd = min(params.N_SGD, model.nD, model.nSe)
        allResValues = []
        for i in range(params.N_ITER):
            optimizer.zero_grad()

            finalX = self.model.forward1(edgeIndex, drugFeatures)
            assert not torch.isnan(finalX).any()

            vd1, vd2, vd3, sampleTrain = self.model.sampleDims2(nsample=nd, toTuple=True)
            # print(len(vd3))
            out = self.model.forward2(finalX, sampleTrain, vd3)
            out = out.reshape(nd, nd, nd)

            targetx = target[vd1, :, :]
            targetx = targetx[:, vd2, :]
            targetx = targetx[:, :, vd3]

            errTrain = self.model.getWLoss(targetx, out)

            assert not torch.isnan(errTrain).any()

            errTrain.backward()
            optimizer.step()
            # self.model.projectNonNegW()

            if i % params.ITER_DB == 0:
                print("\r@Iter ", i, end=" ")
                with torch.no_grad():
                    outTest = self.model.forward3(finalX, testPairs, testAnchor).cpu().detach().numpy()
                    outValid = self.model.forward3(finalX, validPairs, validAnchor).cpu().detach().numpy()

                    auc, aupr = evalAUCAUPROrigin(outTest, testLabel.cpu().numpy())
                    arAUCAUPR.append((auc, aupr))

                    aucv, auprv = evalAUCAUPROrigin(outValid, validLabel.cpu().numpy())
                    arAUCVal.append(auprv)
                    cTime = time.time()
                    self.logger.infoAll((auc, aucv, aupr, auprv, "Elapse@: ", i, cTime - startTime))

                    reSec = []
                    for kk in range(params.N_SEC):
                        indice = segTests[kk]
                        outk = outTest[indice]
                        targetk = testLabel.cpu().numpy()[indice]
                        auck, auprk = evalAUCAUPROrigin(outk, targetk)

                        reSec.append([auck, auprk])
                        if kk == params.N_SEC - 1:
                            allResValues.append([outk, targetk])
                    arSecs.append(reSec)

        selectedInd = np.argmax(arAUCVal)
        auc, aupr = arAUCAUPR[selectedInd]
        vv = -1
        if params.ON_REAL:
            vv = arSecs[selectedInd]
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


def evalAUCAUPROrigin(out, target):
    if np.sum(target) == 0 or np.sum(target) == target.shape[-1]:
        return 0.5, 0.5
    aupr = average_precision_score(target, out)
    auc = roc_auc_score(target, out)
    return auc, aupr


if __name__ == "__main__":
    pass

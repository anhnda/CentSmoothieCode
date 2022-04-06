import random

from models.hegnn.hegnn import HEGNN

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


class WrapperHEGNN:
    def __init__(self, device=torch.device('cpu')):

        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.name = "HEGNN"
        self.isFitAndPredict = True

    def setLogger(self, logger):
        self.logger = logger
        self.logger.infoAll(inspect.getsource(HEGNN))

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

    def loadNegIds(self, ses, secs, tpls, segId=0):
        secs = secs[::-1]
        sz = len(secs)
        indices = [[] for _ in range(sz)]
        dSeId2Tpls = dict()

        dSeId2Indices = dict()

        ses2 = set(ses)

        for seOfId in ses:
            dSeId2Tpls[seOfId] = []
            dSeId2Indices[seOfId] = []
        for ii, tpl in enumerate(tpls):
            _, _, adrIdOf = tpl
            i = sz
            for i, sec in enumerate(secs):
                if adrIdOf in sec:
                    break

            for j in range(i, sz):
                indices[j].append(ii)
        selectedIndices1 = indices[segId]
        for idx in selectedIndices1:
            tpt = tpls[idx]
            _, _, adrIdOf = tpt
            if adrIdOf in ses2:
                dSeId2Tpls[adrIdOf].append(tpt)
                dSeId2Indices[adrIdOf].append(idx)

        for k, v in dSeId2Tpls.items():
            dSeId2Tpls[k] = np.asarray(v)

        return dSeId2Tpls, dSeId2Indices

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

    def sampleRandomWalk(self, adjacency, drugOffset, toTensor=True, device=None):
        nNodes = len(adjacency)
        randomWalkList = [[[], []] for _ in range(nNodes)]
        maxNeighbor = 100
        maxDrugNeighbor = 70
        # print("N nodes: ", nNodes, drugOffset)
        for iNode in range(nNodes):
            # print(iNode)
            nNeighbor = 0
            nDrugNeighbor = 0
            currentNode = iNode
            if len(adjacency[currentNode]) == 0:
                print("No neighbor for: ", currentNode)
                continue
            # print(adjacency[currentNode])
            while nNeighbor < maxNeighbor:
                p = random.random()
                if p > 0.5:
                    currentNeighbors = adjacency[currentNode]
                    if len(currentNeighbors) > 0:
                        sampledNode = random.choice(currentNeighbors)
                        currentNode = sampledNode

                        nodeType = 0  # Default to drug
                        if sampledNode >= drugOffset:
                            nodeType = 1  # SE
                        if nodeType == 0:
                            if nDrugNeighbor < maxDrugNeighbor:
                                nDrugNeighbor += 1
                            else:
                                continue
                        randomWalkList[iNode][nodeType].append(sampledNode)

                        nNeighbor += 1

                else:
                    currentNode = iNode
        # print("Next")
        sampledNodeList = [[[], []] for _ in range(nNodes)]
        sampledNodeSize = []
        for iNode in range(nNodes):
            neighbors = randomWalkList[iNode]
            topKs = [10, 3]
            nSize = []

            for i in range(2):
                neighbor = neighbors[i]
                topK = topKs[i]
                cs = Counter(neighbor)
                topList = cs.most_common(topK)
                sz = len(topList)
                for j in range(sz):
                    sampledNodeList[iNode][i].append(topList[j][0])
                random.shuffle(sampledNodeList[iNode][i])

                nSize.append(sz)
            sampledNodeSize.append(nSize)
        if toTensor:
            drugNeighborList = []
            seNeighborList = []
            for i in range(nNodes):
                drugNeighborList.append(sampledNodeList[i][0])
                seNeighborList.append(sampledNodeList[i][1])
            drugNeighborList = torch.from_numpy(np.asarray(drugNeighborList)).long()
            seNeighborList = torch.from_numpy(np.asarray(seNeighborList)).long()
            if device is not None:
                drugNeighborList = drugNeighborList.to(device)
                seNeighborList = seNeighborList.to(device)
            sampledNodeList = (drugNeighborList, seNeighborList)

        return sampledNodeList, sampledNodeSize


    def train(self, dataWrapper, iFold, method="New", printDB=params.PRINT_DB):
        realData = dataWrapper.data
        target = dataWrapper.ddiTensorInDevice
        model = HEGNN(realData.featureSize, params.EMBEDDING_SIZE, realData.nSe, realData.nD, device=self.device)

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

        if params.EXPORT_TOP_NEG:
            print("Exporting TOP NEGs...")
            dAdrName, dDrugName = utils.load_obj(params.ID2NamePath)
            outFile = "%s/Export_TOP_NEG_%s_%s" % (params.OUTPUT_DIR, method, iFold)
            predictedValues = utils.load_obj("%s/SaveCalValues_W_%s_%s" % (params.OUTPUT_DIR, method, iFold))
            _, outNegK = predictedValues
            ses = secsList[-1][-50:]
            dSeId2Tpls, dSeId2Indices = self.loadNegIds(ses, secs, negTestTpl, segId=-1)
            self.exportTopNeg(dSeId2Tpls, dSeId2Indices, outNegK, realData.nD, dAdrName, dDrugName, outFile)
            exit(-1)

        if params.VISUAL:

            print("Visualize: ...")

            dataX = np.loadtxt("%s%sW_%s" % (params.EMBEDDING_PREX, method, iFold))
            wx = np.loadtxt("%s%sW_Weight%s" % (params.EMBEDDING_PREX, method, iFold))
            dADR2Name, dDrug2Name = utils.load_obj(params.ID2NamePath)
            print(len(dADR2Name), len(dDrug2Name))
            dName2DrugId = utils.reverse_dict(dDrug2Name)
            drugNamePairs = [[["Diazepam", "Clarithromycin"]],
                             [["Hydroxyzine", "Warfarin"]],
                             [["Simvastatin", "Glipizide"]],
                             [["Prednisone", "Tolterodine"]]]
            seIds = sortedADRs[-5:]
            drungIdPairs = []
            for pList in drugNamePairs:
                vv = []
                for p in pList:
                    d1, d2 = p
                    vv.append([dName2DrugId[d1], dName2DrugId[d2]])
                drungIdPairs.append(vv)

            for ii, seId in enumerate(seIds):
                w = wx[seId]
                dataXW = dataX * np.tile(w, (dataX.shape[0], 1))
                self.visual2(trainTpl, seId, realData.nD, dataXW, method, [], iFold, dADR2Name, dDrug2Name)
            exit(-1)
            return 0, 0, 0



        # trainIds = torch.from_numpy(np.asarray(trainTpl)).long().to(self.device)
        testIds = torch.from_numpy(np.asarray(testTpl)).long().to(self.device)
        validIds = torch.from_numpy(np.asarray(validTpl)).long().to(self.device)

        adjacency = dataWrapper.heterogeneousAdjacency

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

            sampledNodeList, sampledNodeSize = self.sampleRandomWalk(adjacency, realData.nD, device=self.device)
            # print(len(sampledNodeList))
            # print(sampledNodeSize)

            # exit(-1)
            # finalX = self.model.forward1(drugFeatures, dd)
            finalX = self.model.forward1(drugFeatures, sampledNodeList)
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
            # self.model.projectNonNegW()

            if i % params.ITER_DB == 0:
                print("\r@Iter ", i, end=" ")
                # print(torch.max(self.model.dimWeightList[0].weight), torch.min(self.model.dimWeightList[0].weight))
                # print(torch.max(self.model.lastDimWeight.weight), torch.min(self.model.lastDimWeight.weight))

                outTest = self.model.forward2(finalX, testIds).cpu().detach().numpy()
                outValid = self.model.forward2(finalX, validIds).cpu().detach().numpy()
                outNegTest = self.model.forward2(finalX, negTestIds).cpu().detach().numpy()

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
            # np.savetxt("%s%sW_%s" % (params.EMBEDDING_PREX, method, iFold), finalX.cpu().detach().numpy())
            # np.savetxt("%s%sW_Weight%s" % (params.EMBEDDING_PREX, method, iFold),
            #            self.model.lastDimWeight.weight.cpu().detach().numpy())
            # predictedValues = allResValues[selectedInd]
            # utils.save_obj(predictedValues, "%s/SaveCalValues_W_%s_%s" % (params.OUTPUT_DIR, method, iFold))

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

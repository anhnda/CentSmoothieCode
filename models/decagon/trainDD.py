from models.decagon import Decagon

import torch
import numpy as np
import inspect
import params
from sklearn.metrics import roc_auc_score, average_precision_score
import time
from utils import utils


def getMSE(a1, a2):
    v = a1 - a2
    v = np.multiply(v, v)
    return np.sqrt(np.sum(v) / (v.shape[0] * v.shape[0]))


class WrapperDecagon:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.name = "Decagon"
        self.isFitAndPredict = True
        if params.FORCE_CPU:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def setLogger(self, logger):
        self.logger = logger
        self.logger.infoAll(inspect.getsource(Decagon))

    def getLoss(self, out1, out2, reg=None):
        # print("OUT MIN MAX FOR LOSS: ", torch.min(out1), torch.min(out2), torch.max(out1), torch.max(out2))
        e1 = - torch.sum(torch.log(out1)) - torch.sum(torch.log(1 - out2))
        if reg is not None:
            if params.R_TYPE == "L1":
                e1 += params.LAMBDA_R * torch.sum(torch.abs(reg))
            else:  # L2:
                e1 += params.LAMBDA_R * torch.sum(torch.mul(reg, reg))
        return e1

    def getLossNP(self, out1, out2):
        e1 = - np.sum(np.log(out1)) - np.sum(np.log(1 - out2))
        return e1

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

    def exportTopNeg(self, dSeId2Tpls, dSeId2Indices, predScores, dADR2Name, dDrug2Name, outFile):

        seOfIds = sorted(dSeId2Indices.keys())
        sorteddSeId2Tpls = dict()
        sortedSeId2Scores = dict()

        for seOfId in seOfIds:
            indices = dSeId2Indices[seOfId]
            tpls = dSeId2Tpls[seOfId]

            res = predScores[indices]
            assert len(res) == len(tpls)

            sortedIndiceScores = np.argsort(res)[::-1]
            assert res[sortedIndiceScores[0]] >= res[sortedIndiceScores[1]]
            # print(res[sortedIndiceScores[0]])
            rr = []
            orr = dSeId2Tpls[seOfId]
            rscore = []
            for idx in sortedIndiceScores:
                d1, d2 = orr[idx]
                rr.append((d1, d2))
                rscore.append(res[idx])
            sorteddSeId2Tpls[seOfId] = rr
            sortedSeId2Scores[seOfId] = rscore
        fout = open(outFile, "w")
        for k, v in sorteddSeId2Tpls.items():
            adrName = dADR2Name[k]
            drugPairs = v
            rrscore = sortedSeId2Scores[k]
            fout.write("%s\n" % adrName)
            for ii, pair in enumerate(drugPairs):
                d1, d2 = pair
                fout.write("\t%s, %s, %s\n" % (dDrug2Name[d1], dDrug2Name[d2], rrscore[ii]))
            fout.write("\n_________\n")

        fout.close()

    def train(self, realData, iFold, method="New", printDB=params.PRINT_DB):
        print("Train: ", realData.nD, realData.nSe)
        self.model = Decagon(featureSize=realData.featureSize, embeddingSize=params.EMBEDDING_SIZE, nSe=realData.nSe,
                             nD=realData.nD, nPro=realData.nPro, device=self.device)
        mseLoss = torch.nn.MSELoss()

        if params.OPTIMIZER == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        else:
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=0.01)

        drugFeatures = torch.from_numpy(realData.drug2Features).float().to(self.device)
        edgeIndex = torch.tensor(realData.ppGraph, dtype=torch.long).t().contiguous().to(self.device)
        # print(edge_index)
        # exit(-1)
        # ids = torch.from_numpy(np.arange(0, realData.nD)).long()

        edge2Label = realData.pTrainPair2Label
        validPosPair2Label = realData.pTrainPair2Label

        testPosPair2Label = realData.pTestPosLabel
        testNegPair2Label = realData.pTestNegLabel
        gA = torch.from_numpy(realData.gA).float()
        gD = realData.gD

        trainPairs, trainAnchor, trainLabel = self.convertPairLabel(edge2Label, testNegPair2Label, realData.nSe)
        validPairs, validAnchor, validLabel = self.convertPairLabel(validPosPair2Label, testNegPair2Label, realData.nSe)
        testPairs, testAnchor, testLabel, labelSegs, dSe2NegPairAll, dSe2NegIndicesAll = self.convertPairLabelWithLabelP(
            testPosPair2Label, testNegPair2Label, realData.nSe)

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

        #
        # if method == "New":
        #     print("In New: Max A: ", np.max(A))
        #     A = torch.from_numpy(A).float()
        # elif method == "Old":
        #     A = torch.from_numpy(UA).float()
        # else:
        #     print("Error: Unknown L method")
        #     exit(-1)
        #
        # if params.DEG_NORM:
        #
        #     if method == "New":
        #         D = torch.from_numpy(D).float()
        #     elif method == "Old":
        #         D = torch.from_numpy(UD).float()
        # else:
        #     D = None

        # A = torch.from_numpy(A).float()
        # randomTpl = syntheticData.genNegEdges(trainTpl)
        arAUCAUPR = []
        arAUCVal = []
        arTrainError = []

        startTime = time.time()
        arSecs = []
        finalX = None
        allRes = []
        for i in range(params.N_ITER):
            optimizer.zero_grad()

            finalX = self.model.forward1(edgeIndex, drugFeatures)
            assert not torch.isnan(finalX).any()
            outTrain = self.model.forward2(finalX, trainPairs, trainAnchor)

            errTrain = mseLoss(outTrain, trainLabel)
            # print(errTrain)
            errTrain.backward()
            optimizer.step()
            if i % params.ITER_DB == 0:
                # continue
                print("\r@Iter ", i, end="")
                # arTrainError.append(errTrain.detach().numpy())
                # if printDB:
                #     self.logger.infoAll(("Error train: ", errTrain,  "Error Validate", errValid, "Error test: ", errTest))
                # print("\t", errTrain.detach().numpy())

                # rout1 = out1.detach().numpy()
                # rout2 = out2.detach().numpy()
                # print("\t O1: ", np.mean(rout1), rout1[:10])
                # print("\t O2: ", np.mean(rout2), rout2[:10])

                # if i == params.N_ITER - 1:
                #    print("Final iteration.")
                outTest = self.model.forward2(finalX, testPairs, testAnchor).cpu().detach().numpy()

                # print("  OUTTEST : ", outTest[:10])
                outValid = self.model.forward2(finalX, validPairs, validAnchor).cpu().detach().numpy()
                # lV = self.getLossNP(outValid, outNegTest)
                # fx = finalX.detach().numpy()
                # drugF = drugFeatures.numpy()
                # d = rData.groupPair2Se
                auc, aupr = evalAUCAUPROrigin(outTest, testLabel.cpu().numpy())
                arAUCAUPR.append((auc, aupr))

                # print("Test AUC, AUPR: ", auc, aupr)
                aucv, auprv = evalAUCAUPROrigin(outValid, validLabel.cpu().numpy())
                arAUCVal.append(auprv)
                cTime = time.time()
                self.logger.infoAll((auc, aucv, auprv, aupr, "Elapse@: ", i, cTime - startTime))

                reSec = []
                for kk in range(params.N_SEC):
                    indice = segTests[kk]
                    outk = outTest[indice]
                    targetk = testLabel.cpu().numpy()[indice]
                    auck, auprk = evalAUCAUPROrigin(outk, targetk)

                    reSec.append([auck, auprk])
                    if kk == params.N_SEC - 1:
                        allRes.append([outk, targetk])
                arSecs.append(reSec)

                from utils.tsne import tsneDrugSe, tsneDrugSe2

        selectedInd = np.argmax(arAUCVal)
        # print(selectedInd, arValError)
        # print(np.argmin(arTrainError), arTrainError)
        auc, aupr = arAUCAUPR[selectedInd]
        vv = arSecs[selectedInd]
        print(vv)

        np.savetxt("%s%s_%s" % (params.EMBEDDING_PREX, "Decagon", iFold), finalX.cpu().detach().numpy())

        # selectedInd = -1
        # self.logger.infoAll(("RE@: ", selectedInd, ": Train, Val, Test: ", trainErrors[selectedInd], validErrors[selectedInd], testErrors[selectedInd]))
        # output = outputs[selectedInd]
        # self.logger.infoAll(("Smooth: ", np.trace(np.dot(output.transpose(), np.dot(syntheticData.XNL, output)))))

        rr = allRes[selectedInd]
        utils.save_obj(rr, "%s/SaveCalValues_%s_%s" % (params.OUTPUT_DIR, "Decagon", iFold))
        return auc, aupr, vv  # testErrors[selectedInd], trainErrors[selectedInd] # getMSE(output, syntheticData.labels)


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
    import random
    from utils import utils
    from dataFactory.polyADR import RealData

    random.seed(params.TORCH_SEED)
    torch.manual_seed(params.TORCH_SEED)
    np.random.seed(params.TORCH_SEED)

    method = "Old"
    rData = utils.load_obj(params.R_DATA)
    model = WrapperGCov()

    iFold = 18
    for i in range(1):
        # for i in range(params.K_FOLD):
        auc, aupr = model.train(rData, i, method)
        print("IFold: ", i, "AUC, AUPR: ", auc, aupr)
    print("Method: ", method)

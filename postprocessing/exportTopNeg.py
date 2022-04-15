from utils import utils
import numpy as np
import params
import torch


def __loadNegIds(ses, secs, tpls, segId=0):
    secs = secs[::-1]
    sz = len(secs)
    indices = [[] for _ in range(sz)]
    dSeId2Tpls = dict()
    dSeId2Indices = dict()

    ses2 = set(ses)

    print("Tpls: ", len(tpls), tpls[0])

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
    print("Num selected: ", len(selectedIndices1))

    for idx in selectedIndices1:
        tpt = tpls[idx]
        _, _, adrIdOf = tpt
        if adrIdOf in ses2:
            dSeId2Tpls[adrIdOf].append(tpt)
            dSeId2Indices[adrIdOf].append(idx)

    for k, v in dSeId2Tpls.items():
        dSeId2Tpls[k] = np.asarray(v)

    return dSeId2Tpls, dSeId2Indices


def exportTopNeg(realData, method, iFold):
    sortedADRs = realData.orderADRIds
    negTestTpl = realData.negFold
    nD = realData.nD
    secs = [set() for _ in range(params.N_SEC)]
    secsList = [[] for _ in range(params.N_SEC)]
    secLength = int(len(sortedADRs) / params.N_SEC)
    for i, v in enumerate(sortedADRs):
        secId = int(i / secLength)
        if secId == params.N_SEC:
            secId = params.N_SEC - 1
        secs[secId].add(v + nD)
        secsList[secId].append(v + nD)
    print("Exporting TOP NEGs...")
    dAdrName, dDrugName = utils.load_obj(params.ID2NamePath_TWOSIDE)
    outFile = "%s/Export_TOP_NEG_%s_%s" % (params.OUTPUT_DIR, method, iFold)
    predictedValues = utils.load_obj("%s/SaveCalValues_W_%s_%s" % (params.OUTPUT_DIR, method, iFold))

    ses = secsList[-1][-50:]
    print("Ses: ", len(ses), ses)
    if method == "Decagon":
        testPairs, testAnchor, testLabel, labelSegs, dSe2NegPairAll, dSe2NegIndicesAll = convertPairLabelWithLabelP(
            realData.pTestPosLabel, realData.pTestNegLabel, realData.nSe)
        predictedScores, _ = predictedValues
        print("Predicted: ", len(predictedScores), predictedScores[0])
        print(predictedScores[:20])
        dSeId2Tpls, dSeId2Indices = __loadNegIdsSe2Pair(ses, dSe2NegPairAll, dSe2NegIndicesAll, seOffset=realData.nD)
        __exportTopNegBySe2Pair(dSeId2Tpls, dSeId2Indices, predictedScores, dAdrName, dDrugName, outFile)
    else:
        _, outNegK = predictedValues
        outNegK = outNegK.reshape(-1)
        print("Neg: ", len(outNegK), outNegK[0])
        print(outNegK[:20])
        dSeId2Tpls, dSeId2Indices = __loadNegIds(ses, secs, negTestTpl, segId=-1)
        __exportTopNeg2(dSeId2Tpls, dSeId2Indices, outNegK, nD, dAdrName, dDrugName, outFile)


def __exportTopNeg2(dSeId2Tpls, dSeId2Indices, NegRes, nD, dADR2Name, dDrug2Name, outFile):
    seOfIds = sorted(dSeId2Indices.keys())
    sorteddSeId2Tpls = dict()
    sortedSeId2Scores = dict()
    print(NegRes[:100])
    for seOfId in seOfIds:
        indices = dSeId2Indices[seOfId]
        tpls = dSeId2Tpls[seOfId]

        res = NegRes[indices]
        print(res[:20])
        assert len(res) == len(tpls)

        sortedIndiceScores = np.argsort(res)[::-1]
        assert res[sortedIndiceScores[0]] >= res[sortedIndiceScores[1]]
        # print(res[sortedIndiceScores[0]])
        rr = []
        orr = dSeId2Tpls[seOfId]
        rscore = []
        for idx in sortedIndiceScores:
            # print(orr[idx])
            d1, d2, _ = orr[idx]
            rr.append((d1, d2))
            rscore.append(res[idx])
        print(rscore)
        sorteddSeId2Tpls[seOfId - nD] = rr
        sortedSeId2Scores[seOfId - nD] = rscore
    fout = open(outFile, "w")
    fseNames = open("%s_se" % outFile, "w")

    for k, v in sorteddSeId2Tpls.items():
        adrName = dADR2Name[k]
        drugPairs = v
        rscore = sortedSeId2Scores[k]
        fout.write("%s\n" % adrName)
        fseNames.write("%s\n" % adrName)
        for ii, pair in enumerate(drugPairs):
            d1, d2 = pair
            fout.write("\t%s, %s, %s\n" % (dDrug2Name[d1], dDrug2Name[d2], rscore[ii]))
        fout.write("\n_________\n")

    fout.close()
    fseNames.close()

def convertPairLabelWithLabelP(ddPos, ddNeg, sz):
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

    return torch.from_numpy(np.asarray(pairs)).long(), torch.from_numpy(
        np.asarray(allAnchor)).long(), \
           torch.from_numpy(np.asarray(trueLabels)).float(), labelSegs, dSe2NegPair, dSe2NegIndices





def __loadNegIdsSe2Pair(ses, dSe2PairAll, dSe2IndicesAll, seOffset=0):
    dSeId2Tpls = dict()

    dSeId2Indices = dict()

    for seId in ses:
        seId -= seOffset
        dSeId2Tpls[seId] = dSe2PairAll[seId]
        dSeId2Indices[seId] = dSe2IndicesAll[seId]

    return dSeId2Tpls, dSeId2Indices


def __exportTopNegBySe2Pair(dSeId2Tpls, dSeId2Indices, predScores, dADR2Name, dDrug2Name, outFile):
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
        # print(rscore)
        sorteddSeId2Tpls[seOfId] = rr
        sortedSeId2Scores[seOfId] = rscore
    fout = open(outFile, "w")
    fseNames = open("%s_se" % outFile, "w")
    for k, v in sorteddSeId2Tpls.items():
        adrName = dADR2Name[k]
        drugPairs = v
        rrscore = sortedSeId2Scores[k]
        fout.write("%s\n" % adrName)
        fseNames.write("%s\n" % adrName)
        for ii, pair in enumerate(drugPairs):
            d1, d2 = pair
            fout.write("\t%s, %s, %s\n" % (dDrug2Name[d1], dDrug2Name[d2], rrscore[ii]))
        fout.write("\n_________\n")

    fout.close()
    fseNames.close()
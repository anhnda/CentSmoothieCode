import numpy as np
import random
from utils import utils
from dataFactory.genData import synParams
import params
from numpy.linalg import eigh
from scipy.stats import bernoulli
import random
import params
from dataFactory.genData.func import print_db, genTrueNegTpl, genAdjacency
import copy
import itertools
from dataFactory.dataLoader import RealData, RealFoldData

from dataFactory.genData.lh import *
from dataFactory.genData.func import trainFold2PairStats
from multiprocessing import Process, Value, Queue
import time

DATASET_DIR = "%s/SynDDI" % params.TMP_DIR
utils.ensure_dir(DATASET_DIR)
DUMP_FILE_PREF = "%s/dump" % DATASET_DIR


class SyntheticData:
    def __init__(self, maxDrug, maxSeg, maxSe, seOffset=False):
        print("Inp: ", maxDrug, maxSeg, maxSe)
        self.maxDrug = maxDrug
        self.maxSE = maxSe
        self.maxPart = maxSeg
        if seOffset:
            self.seIdOffset = maxDrug
        else:
            self.seIdOffset = 0
        if self.maxSE == -1:
            self.maxSE = int(maxSeg * (maxSeg - 1) / 2)
        print("SYN MAX SE: ", maxSe, self.maxSE)
        print("M_LIM_SEG", synParams.M_LIM_SEG)

    def genMeans(self, segs, sz):
        meanx = np.zeros(sz)
        for segId in segs:
            meanx[segId * synParams.SUBSIZE: (1 + segId) * synParams.SUBSIZE].fill(1)
        return meanx

    def genDrugF(self):
        # Segment
        drugId2Segs = dict()
        segList = [i for i in range(self.maxPart)]
        self.seg2drugId = dict()
        re = []

        for i in range(1, synParams.M_MAX_SEG + 1):
            v = np.random.randint(1, i + 1, self.maxDrug)
            re.append(v)
        re = np.vstack(re).transpose()

        maxSizeSample = []
        for dId in range(self.maxDrug):
            s = random.sample(segList, synParams.M_MAX_SEG)
            maxSizeSample.append(s)
        print(re.shape)
        for dId in range(self.maxDrug):
            nSeg = re[dId, synParams.M_LIM_SEG - 1]
            segs = maxSizeSample[dId][:nSeg]
            drugId2Segs[dId] = segs
            for seg in segs:
                drugIds = utils.get_insert_key_dict(self.seg2drugId, seg, [])
                drugIds.append(dId)
        self.drugId2Segs = drugId2Segs

        # Features:
        self.featureSize = synParams.SUBSIZE * self.maxPart
        noNoiSize = self.featureSize
        if synParams.IS_NOISE:
            self.featureSize += synParams.NOISESIZE

        drug2Features = dict()

        COV = np.zeros(noNoiSize)
        COV.fill(synParams.COV)
        COV = np.diag(COV)
        drugFs = []
        for dId in range(self.maxDrug):
            segs = drugId2Segs[dId]
            means = self.genMeans(segs, noNoiSize)
            f = np.random.multivariate_normal(means, COV, 1)
            drug2Features[dId] = f
            drugFs.append(f)

        drugFs = np.vstack(drugFs)

        if synParams.IS_NOISE:
            noiseMean = np.zeros(synParams.NOISESIZE)
            noiseMean.fill(1)
            COV = np.zeros(synParams.NOISESIZE)
            COV.fill(synParams.NOISE_COV)
            COV = np.diag(COV)
            noiseFeatures = np.random.multivariate_normal(noiseMean, COV, self.maxDrug)
            drugFs = np.concatenate((drugFs, noiseFeatures), axis=1)

        drugFs[drugFs < 0] = 0
        self.drugFeatures = drugFs

    def genSe(self):
        segSet = set()
        segList = [i for i in range(self.maxPart)]
        pairSet = set()

        se2PairSegs = dict()
        pairSeg2Se = dict()

        def samplePair():
            g1, g2 = random.sample(segList, 2)
            g1, g2 = swap(g1, g2)
            return g1, g2

        for i in range(self.maxSE):
            # Sample 2 groups
            g1, g2 = samplePair()
            nextPair = (g1, g2)

            if nextPair in pairSet:
                g1, g2 = samplePair()
                nextPair = (g1, g2)
                while nextPair in pairSet:
                    g1, g2 = samplePair()
                    nextPair = (g1, g2)

            pairSet.add((g1, g2))
            segSet.add(g1)
            segSet.add(g2)

            seId = len(se2PairSegs)
            se2PairSegs[seId] = nextPair
            pairSeg2Se[nextPair] = seId

        print(list(pairSet))
        self.se2PairSegs = se2PairSegs

    def genAllPosEges(self):

        allTuples = []
        se2DrugPair = dict()
        drugSet = set()
        dRemapDrug = dict()
        tupleSet = set()
        allPairDrugs = set()
        validG = set()
        for seId in range(self.maxSE):
            g1, g2 = self.se2PairSegs[seId]
            validG.add(g1)
            validG.add(g2)
            pairDrugSet = set()
            se2DrugPair[seId] = pairDrugSet
            d1s = self.seg2drugId[g1]
            d2s = self.seg2drugId[g2]
            for d1 in d1s:
                for d2 in d2s:
                    if d1 != d2:
                        rd1 = utils.get_update_dict_index(dRemapDrug, d1)
                        rd2 = utils.get_update_dict_index(dRemapDrug, d2)
                        rd1, rd2 = swap(rd1, rd2)

                        if (rd1, rd2) not in pairDrugSet:
                            pairDrugSet.add((rd1, rd2))
                            allPairDrugs.add((rd1, rd2))
                            tlp = (rd1, rd2, seId)
                            allTuples.append(tlp)

        reMapAllTuple = []
        self.dRemapDrug = dRemapDrug
        self.maxDrug = len(dRemapDrug)
        if self.seIdOffset > 0:
            self.seIdOffset = self.maxDrug

        for tlp in allTuples:
            d1, d2, seId = tlp
            rSeId = seId + self.seIdOffset
            rTlp = d1, d2, rSeId
            reMapAllTuple.append(rTlp)

        self.allTuples = reMapAllTuple
        self.se2DrugPair = se2DrugPair

        dDrugNew2Old = utils.reverse_dict(dRemapDrug)
        rDrugFs = []

        validFeatures = np.zeros(self.featureSize)
        for g in validG:
            for i in range(g * synParams.SUBSIZE, (g + 1) * synParams.SUBSIZE):
                validFeatures[i] = 1
        for dId in range(self.maxDrug):
            oldId = dDrugNew2Old[dId]
            f = self.drugFeatures[oldId]
            if synParams.ONLY_VALID_SEG:
                f = f * validFeatures
            rDrugFs.append(f)
        rDrugFs = np.vstack(rDrugFs)
        self.drugFeatures = rDrugFs

        return self.allTuples, allPairDrugs

    def genAll(self):
        print("Gen drug features")
        self.genDrugF()
        print("Gen Ses")
        self.genSe()

        print("Gen triples")
        self.genAllPosEges()
        print("Saving...")
        data = (self.drugFeatures, self.allTuples)
        utils.save_obj(data, "%s/fullSyn.dat" % DATASET_DIR)


def genData(nDrug, nMaxSeg, nMaxSe):
    random.seed(params.TORCH_SEED)
    np.random.seed(params.TORCH_SEED)
    params.ON_REAL = False
    params.ON_MG = True
    params.DEG_NORM = False

    mData = SyntheticData(nDrug, nMaxSeg, nMaxSe)

    mData.genAll()


def createSubSet(ratio=0.1):
    utils.resetRandomSeed()

    drugFeatures, allTriples = utils.load_obj("%s/fullSyn.dat" % DATASET_DIR)

    drugSet = set()
    adrSet = set()

    # for t in allTriples:
    #     d1, d2, s = t
    #     drugSet.add(d1)
    #     drugSet.add(d2)
    #     adrSet.add(s)

    triples = random.sample(allTriples, int(ratio * len(allTriples)))
    dADR2Pair = dict()

    for t in triples:
        d1, d2, s = t
        drugSet.add(d1)
        drugSet.add(d2)
        adrSet.add(s)

        pairList = utils.get_insert_key_dict(dADR2Pair, s, [])
        pairList.append((d1, d2))

    orderedADR = [i for i in range(len(adrSet))]
    drug2Fingerprint = dict()
    for i in range(len(drugSet)):
        drug2Fingerprint[i] = drugFeatures[i]
    v = (len(adrSet), len(drugSet), dADR2Pair, orderedADR, drug2Fingerprint)

    print(sorted(drugSet))
    print(sorted(adrSet))
    print(len(drugSet), len(adrSet))
    print(len(triples) * 2 / (len(drugSet) * len(drugSet) * len(adrSet)))

    utils.save_obj(v, "%s_%1.1f" % (DUMP_FILE_PREF, ratio))


def genHyperData(ratio):
    nADR, nDrug, dADR2Pair, orderedADR, inchi2FingerPrint = utils.load_obj("%s_%1.1f" % (DUMP_FILE_PREF, ratio))
    print_db(nADR, len(dADR2Pair), nDrug, len(inchi2FingerPrint))

    dADR2Id = dict()
    dInchi2Id = dict()
    dADRId2PairIds = dict()

    adrs = sorted(list(dADR2Pair.keys()))
    allPairs = set()
    orderedADRIds = list()
    for adr in adrs:
        adrId = utils.get_update_dict_index(dADR2Id, adr)
        pairs = dADR2Pair[adr]
        for pair in pairs:
            inchi1, inchi2 = pair
            d1 = utils.get_update_dict_index(dInchi2Id, inchi1)
            d2 = utils.get_update_dict_index(dInchi2Id, inchi2)
            d1, d2 = swap(d1, d2)
            pairIds = utils.get_insert_key_dict(dADRId2PairIds, adrId, set())
            pairIds.add((d1, d2))
            allPairs.add((d1, d2))
    for oADr in orderedADR:
        adrId = dADR2Id[oADr]
        orderedADRIds.append(adrId)
    print_db("Drug, ADR, Pairs: ", len(dInchi2Id), len(adrs), len(allPairs))
    print_db("Loading ADR 2 Pair completed")

    numDrug = len(dInchi2Id)
    numSe = len(dADR2Id)
    numNodes = numDrug + numSe
    print_db(numDrug, numSe, numNodes)
    # Create Feature Matrix:

    dDrugId2Inchi = utils.reverse_dict(dInchi2Id)

    features = []
    inchies = []
    for i in range(numDrug):
        inchi = dDrugId2Inchi[i]
        inchies.append(inchi)
        fs = inchi2FingerPrint[inchi]
        features.append(fs)
    features = np.asarray(features)
    if params.ONE_HOT:
        nD = features.shape[0]
        features = np.diag(np.ones(nD))
    print_db("Feature: ", features.shape)


    negFold = genTrueNegTpl(dADRId2PairIds, numDrug, params.SAMPLE_NEG)
    print("Starting...")
    for iFold in range(params.K_FOLD):
        print("Gen fold: ", iFold)
        data = dADRId2PairIds, numDrug, numNodes, iFold, numSe, negFold, features, None, [], 0, orderedADRIds
        realFold = producer(data)
        print("Saving fold: ", iFold)

        utils.save_obj(realFold, "%s/S_%d_%d_%1.1f_%d" % (
            DATASET_DIR, synParams.M_MAX_SE, synParams.M_MAX_DRUG, ratio, iFold))



def producer(data):
    dADRId2PairIds, numDrug, numNodes, iFold, numSe, negFold, features, smiles, edgeIndex, nProtein, orderedADRIds = data
    testFold = []
    trainFold = []
    validFold = []

    edgeSet = set()
    edge2Label = dict()

    for adr, pairs in dADRId2PairIds.items():

        adr = adr + numDrug
        pairs = sorted(list(pairs))
        pairs = copy.deepcopy(pairs)
        random.seed(params.TORCH_SEED)
        random.shuffle(pairs)
        nSize = len(pairs)
        foldSize = int(nSize / params.K_FOLD)
        startTest = iFold * foldSize
        endTest = (iFold + 1) * foldSize
        if endTest > nSize:
            endTest = nSize

        if iFold == params.K_FOLD - 1:
            startValid = 0
        else:
            startValid = endTest

        endValid = startValid + foldSize

        for i in range(nSize):
            d1, d2 = pairs[i]
            tpl = (d1, d2, adr)

            if startTest <= i < endTest:
                testFold.append(tpl)
            elif startValid <= i < endValid:
                validFold.append(tpl)

            else:
                trainFold.append(tpl)
                edgeSet.add((d1, d2))
                labels = utils.get_insert_key_dict(edge2Label, (d1, d2), [])
                labels.append(adr - numDrug)

    pairStats = trainFold2PairStats(trainFold, numDrug)

    testPosPair2Label = dict()
    validPosPair2Label = dict()
    testNegPair2Label = dict()

    for tpl in testFold:
        d1, d2, adr = tpl
        posLabels = utils.get_insert_key_dict(testPosPair2Label, (d1, d2), [])
        posLabels.append(adr - numDrug)

    for tpl in validFold:
        d1, d2, adr = tpl
        posLabels = utils.get_insert_key_dict(validPosPair2Label, (d1, d2), [])
        posLabels.append(adr - numDrug)

    for tpl in negFold:
        d1, d2, adr = tpl
        negLabels = utils.get_insert_key_dict(testNegPair2Label, (d1, d2), [])
        negLabels.append(adr - numDrug)

    for edge in edgeSet:
        d1, d2 = edge
        edgeIndex.append([d1, d2])
        edgeIndex.append([d2, d1])



    heterogeneousAdjacency = genAdjacency(trainFold, numDrug + numSe)

    A, D = genAFromTpl(trainFold, numNodes)
    UA, UD = genUAFromTpl(trainFold, numNodes)

    realFold = RealFoldData(trainFold, testFold, validFold, A, UA, negFold, features)
    realFold.nSe = numSe
    realFold.nD = numDrug
    realFold.DFold = D
    realFold.UDFold = UD

    realFold.trainPairStats = pairStats
    realFold.iFold = iFold

    realFold.pEdgeSet = edgeSet
    realFold.pTrainPair2Label = edge2Label
    realFold.pValidPosLabel = validPosPair2Label
    realFold.pTestPosLabel = testPosPair2Label
    realFold.pTestNegLabel = testNegPair2Label
    realFold.dADR2Drug = dADRId2PairIds
    realFold.batchSMILE = smiles
    realFold.ppGraph = edgeIndex
    realFold.nPro = nProtein
    realFold.orderADRIds = orderedADRIds

    realFold.heterogeneousAdjacency = heterogeneousAdjacency
    return realFold

def runGenFullData():
    utils.resetRandomSeed()
    genData(synParams.M_MAX_DRUG, synParams.M_MAX_SEG, synParams.M_MAX_SE)


# def checkSubSetCreation():
#    createSubSet()

def genKFoldByRatio():
    for ratio in [0.1 * i for i in range(3,11)]:
        print(ratio)
        utils.resetRandomSeed()
        createSubSet(ratio)
        genHyperData(ratio)

if __name__ == "__main__":
    # runGenFullData()
    genKFoldByRatio()
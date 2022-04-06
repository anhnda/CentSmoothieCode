import dataFactory.dataLoader
import params
from utils import utils
import random
import copy
from dataFactory.moleculeFactory import  MoleculeFactory
import itertools
from multiprocessing import Process, Value, Queue
from dataFactory.dataLoader import RealData, RealFoldData
from dataFactory.genData.func import *
import numpy as np
import torch

PATH = params.PATH_JADERDDI
MIN_ADR = 60
MIN_DRUG = 5
params.SAMPLE_NEG = 200  # Proportion to the number of positive samples for each drug pair of the dataset


DATASET_DIR = "%s/JADERDDI" % params.TMP_DIR
utils.ensure_dir(DATASET_DIR)
D_PREF = "J"
DUM_FILE = "%s/Dump_%s.pkl" % (DATASET_DIR, D_PREF)

def createSubSet():
    inchi2FingerPrint = loadPubChem()

    inchiKeys = inchi2FingerPrint.keys()

    monoADR = loadMonoADR()
    fin = open(PATH)
    drugSet = set()
    adrSet = set()
    drugCount = dict()
    adrCount = dict()
    drugPair2ADR = dict()
    inchi2Drug = dict()

    while True:
        line = fin.readline()
        if line == "":
            break
        line = line.strip()
        parts = line.split("|")

        d1, d2, inchi1, inchi2 = parts[0], parts[1], parts[2], parts[3]
        if inchi1 not in inchiKeys or inchi2 not in inchiKeys:
            continue
        adrs = parts[4].split(",")
        inchi2Drug[inchi1] = d1
        inchi2Drug[inchi2] = d2
        drugSet.add(inchi1)
        drugSet.add(inchi2)
        utils.add_dict_counter(drugCount, inchi1)
        utils.add_dict_counter(drugCount, inchi2)
        adr1 = utils.get_dict(monoADR, inchi1, set())
        adr2 = utils.get_dict(monoADR, inchi2, set())
        for adr in adrs:
            if adr in adr1 or adr in adr2:
                continue
            adrSet.add(adr)
            utils.add_dict_counter(adrCount, adr)
        drugPair2ADR[(inchi1, inchi2)] = adrs

    fin.close()

    adrCountsSorted = utils.sort_dict(adrCount)
    print_db(adrCountsSorted)
    cc = []
    for p in adrCountsSorted:
        _, v = p
        cc.append(v)


    # from postprocessing import plotHist
    # plotHist.plotHist(cc, 20, "../figs/Hist")
    # plotHist.plotHist(cc, 20, "../figs/Hist5000", 5000)
    # plotHist.plotHist(cc, 20, "../figs/Hist500", 500)


    validADRs = set()
    endADR = min(len(adrCountsSorted), params.ADR_OFFSET + params.MAX_R_ADR)
    orderedADR = list()
    for i in range(params.ADR_OFFSET, endADR):
        adr, v = adrCountsSorted[i]
        if v < MIN_ADR:
            continue
        validADRs.add(adr)
        orderedADR.append(adr)
    print_db("NADR: ", len(validADRs))
    drugCountSorted = utils.sort_dict(drugCount)
    validInchi = set()
    m = min(len(drugCount), params.MAX_R_DRUG)
    for i in range(m):
        inchi, v = drugCountSorted[i]
        if v < MIN_DRUG:
            continue
        validInchi.add(inchi)
    print_db("NDrug: ", len(validInchi))
    dADR2Pair = dict()
    # Filter by ADRs
    allPairs = set()
    s1 = 0
    for pairs, adrs in drugPair2ADR.items():
        inchi1, inchi2 = pairs
        if inchi1 in validInchi and inchi2 in validInchi:
            for adr in adrs:
                if adr in validADRs:
                    pairs = utils.get_insert_key_dict(dADR2Pair, adr, [])
                    pairs.append((inchi1, inchi2))
                    allPairs.add((inchi1, inchi2))
                    s1 += 1
    print_db("Saving ", s1)
    print_db(len(allPairs), len(drugPair2ADR), len(dADR2Pair))
    v = (params.MAX_R_ADR, params.MAX_R_DRUG, dADR2Pair, orderedADR, inchi2FingerPrint)
    utils.save_obj(v, DUM_FILE)
    return v


def stats():
    drugSet, adrSet, drugCount, adrCount, drugPair2ADR, _ = createSubSet()
    v1 = utils.sort_dict(drugCount)
    v2 = utils.sort_dict(adrCount)
    print_db(v1)
    print_db(v2)
    print_db(len(drugSet), len(adrSet), len(drugPair2ADR))


def filter():
    pass

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

    def getPairTypeById(id1, id2, anchor):
        # Drug-Drug: 0, Drug-Side effect: 1, Side effect - Drug: 2, Drug-self-loop: 3, Side effect self loop: 4
        tp = 0
        if id1 < anchor <= id2:
            tp = 1
        elif id1 >= anchor > id2:
            tp = 2
        elif id1 == id2:
            if id1 < anchor:
                tp = 3
            else:
                tp = 4
        return tp
    # hyperEdgeCliqueIndex = []
    # hyperedgeIndexType = []
    # for tpl in trainFold:
    #     for pair in list(itertools.product(tpl, tpl)):
    #         id1, id2 = pair
    #         if id1 != id2:
    #             tp = getPairTypeById(id1, id2, numDrug)
    #             hyperEdgeCliqueIndex.append([id1, id2])
    #             hyperedgeIndexType.append(tp)
    #
    # # Adding self-loop
    # for idx in range(numDrug + numSe):
    #     id1 = idx
    #     tp = getPairTypeById(id1, id1, numDrug)
    #     hyperEdgeCliqueIndex.append([id1, id1])
    #     hyperedgeIndexType.append(tp)

    heterogeneousAdjacency = genAdjacency(trainFold, numDrug + numSe)

    realFold = RealFoldData(trainFold, testFold, validFold, 0, 0, negFold, features)
    realFold.nSe = numSe
    realFold.nD = numDrug
    realFold.DFold = 0

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

    # realFold.hyperEdgeCliqueIndex = torch.tensor(np.asarray(hyperEdgeCliqueIndex), dtype=torch.long).t().contiguous()
    # realFold.hyperEdgeIndexType = np.asarray(hyperedgeIndexType, dtype=int)
    realFold.heterogeneousAdjacency = heterogeneousAdjacency
    return realFold





def genBatchAtomGraph(smiles):
    moleculeFactory = MoleculeFactory()
    for smile in smiles:
        moleculeFactory.addSMILE(smile)
    graphBatch = moleculeFactory.createBatchGraph(atomOffset=0)
    return graphBatch



def genHyperData():
    nADR, nDrug, dADR2Pair, orderedADR, inchi2FingerPrint = utils.load_obj(DUM_FILE)
    print_db(nADR, len(dADR2Pair), nDrug, len(inchi2FingerPrint))

    # Convert 2 Id
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
    allInchies = dDrugId2Inchi.keys()

    features = []
    inchies = []
    for i in range(numDrug):
        inchi = dDrugId2Inchi[i]
        inchies.append(inchi)
        fs = inchi2FingerPrint[inchi]
        features.append(fs)


    smiles = genSMILESFromInchies(inchies)

    edgeIndex, protein2Id, nDrug, dDrug2ProteinFeatures = loadDrug2Protein(inchies)

    appendProteinProtein(protein2Id, edgeIndex, nDrug)

    nProtein = len(protein2Id)
    features = np.vstack(features)

    if params.PROTEIN_FEATURE:
        pFeatures = []
        cc = 0
        for inchi in inchies:
            try:
                ff = dDrug2ProteinFeatures[inchi]
            except:
                ff = np.zeros(nProtein)
                print_db("No Protein for: ", inchi)
                cc += 1
            pFeatures.append(ff)
        print_db("Missing: ", cc)
        pFeatures = np.vstack(pFeatures)
        features = np.concatenate((features, pFeatures), axis=1)
    if params.ONE_HOT:
        nD = features.shape[0]
        features = np.diag(np.ones(nD))

    negFold = genTrueNegTpl(dADRId2PairIds, numDrug, params.SAMPLE_NEG)
    print_db("Starting...")
    for iFold in range(params.K_FOLD):
        print("Gen fold: ", iFold)
        edgeIndex2 = copy.deepcopy(edgeIndex)
        data = dADRId2PairIds, numDrug, numNodes, iFold, numSe, negFold, features, smiles, edgeIndex2, nProtein, orderedADRIds
        realFold = producer(data)
        print("Saving fold: ", iFold)
        utils.save_obj(realFold, "%s/%s_%d_%d_%d_%d" % (
            DATASET_DIR, D_PREF, params.MAX_R_ADR, params.MAX_R_DRUG, params.ADR_OFFSET, iFold))



def exportData():
    params.ON_REAL = True
    params.DEG_NORM = True
    print_db("DRUG, ADR: ", params.MAX_R_DRUG, params.MAX_R_ADR)

    createSubSet()
    genHyperData()


def run():

    print_db("ONEHOT: ", params.ONE_HOT)
    exportData()
    pass
if __name__ == "__main__":
    PRINT_DB = True
    run()
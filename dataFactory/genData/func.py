import numpy as np
import torch

import dataFactory.dataLoader
import params
from dataFactory.moleculeFactory import MoleculeFactory
from utils import utils


def resetRandomSeed():
    import random
    random.seed(params.TORCH_SEED)
    torch.manual_seed(params.TORCH_SEED)
    np.random.seed(params.TORCH_SEED)


def loadPubChem():
    return utils.load_obj(params.PUBCHEM_FILE)


def loadMonoADR():
    fin = open(params.MONO_ADR_FILE)
    dDrug2ADRSet = dict()
    while True:
        line = fin.readline()
        if line == "":
            break
        line = line.strip()
        parts = line.split("|")
        inchi = parts[1]
        adrs = parts[2]
        adrSet = set()
        for adr in adrs:
            adrSet.add(adr)
        dDrug2ADRSet[inchi] = adrSet
    fin.close()
    return dDrug2ADRSet


def loadProtein2Pathway():
    fin = open("%s/KEGG/uniprot_2_pathway.txt" % params.DATA_DIR)
    dPathway2Id = dict()
    dProtein2Pathways = dict()
    while True:
        line = fin.readline()
        if line == "":
            break
        parts = line.strip().split("|")
        protein = parts[0]
        pathways = parts[1].split(",")
        ar = [utils.get_update_dict_index(dPathway2Id, pathway) for pathway in pathways]
        dProtein2Pathways[protein] = ar
    fin.close()
    dPathway2Name = dict()
    fin = open("%s/KEGG/path:hsa.txt" % params.DATA_DIR)
    while True:
        line = fin.readline()
        if line == "":
            break
        parts = line.strip().split("\t")
        dPathway2Name[parts[0]] = parts[1]
    fin.close()
    return dProtein2Pathways, dPathway2Id, dPathway2Name


def loadDrug2Protein(inchies, pathway=True):
    dInchi2Id = dict()
    for inchi in inchies:
        utils.get_update_dict_index(dInchi2Id, inchi)
    nDrug = len(dInchi2Id)
    drug2ProteinList = dataFactory.dataLoader.loadDrugProteinMap()
    # print(drug2ProteinList['ILVYCEVXHALBSC-OTBYEXOQSA-N'])
    proteinListList = sorted(list(drug2ProteinList.values()))
    protensSets = set()
    protein2Id = dict()
    dProtein2Pathways, dPathway2Id, _ = loadProtein2Pathway()

    for proteins in proteinListList:
        for protein in proteins:
            if protein != "":
                protensSets.add(protein)

    proteinList = list(protensSets)
    proteinList = sorted(proteinList)
    for protein in proteinList:
        utils.get_update_dict_index(protein2Id, protein)

    dDrug2ProteinFeatures = dict()
    nP = len(protein2Id)
    nA = len(dPathway2Id)
    nS = nP
    if pathway:
        nS += nA
    edge_index = []
    cc = 0
    for drugInchi, proteins in drug2ProteinList.items():
        drugId = utils.get_dict(dInchi2Id, drugInchi, -1)
        if drugId == -1:
            cc += 1
            continue
        proteinFeature = np.zeros(nS)
        for p in proteins:
            if p == "":
                continue
            piD0 = protein2Id[p]
            proteinFeature[piD0] = 1
            if pathway:
                pa = utils.get_dict(dProtein2Pathways, p, [])
                for a in pa:
                    proteinFeature[a + nP] = 1

            pId = piD0 + nDrug
            edge_index.append([drugId, pId])
            edge_index.append([pId, drugId])
        dDrug2ProteinFeatures[drugInchi] = proteinFeature
    return edge_index, protein2Id, nDrug, dDrug2ProteinFeatures


def appendProteinProtein(protein2Id, edg_index, nDrug):
    fin = open(params.PPI_FILE)
    while True:
        line = fin.readline()
        if line == "":
            break
        parts = line.strip().split("\t")
        p1 = utils.get_dict(protein2Id, parts[0], -1)
        p2 = utils.get_dict(protein2Id, parts[1], -1)
        if p1 != -1 and p2 != -1:
            edg_index.append([p1 + nDrug, p2 + nDrug])
            edg_index.append([p2 + nDrug, p1 + nDrug])

    fin.close()
    return edg_index


def loadInchi2SMILE():
    f = open(params.DRUGBANK_ATC_INCHI)
    inchi2SMILE = dict()
    while True:
        line = f.readline()
        if line == "":
            break
        parts = line.strip().split("\t")
        inchi2SMILE[parts[-1]] = parts[4]
    f.close()
    return inchi2SMILE


def swap(d1, d2):
    if d1 > d2:
        d1, d2 = d2, d1
    return d1, d2


def genBatchAtomGraph(smiles):
    moleculeFactory = MoleculeFactory()
    for smile in smiles:
        moleculeFactory.addSMILE(smile)
    graphBatch = moleculeFactory.createBatchGraph(atomOffset=0)
    return graphBatch


def genSMILESFromInchies(inchies):
    inchi2SMILE = loadInchi2SMILE()
    allSMILEs = []
    for inchi in inchies:
        smile = inchi2SMILE[inchi]
        allSMILEs.append(smile)
    return allSMILEs


def print_db(*msg):
    if params.PRINT_DB:
        print(*msg)


def trainFold2PairStats(trainFold, nOffDrug):
    dii = dict()
    dij = dict()
    dit = dict()
    dtt = dict()
    for tpl in trainFold:
        i, j, t = tpl
        to = t - nOffDrug
        i, j = swap(i, j)

        vdii = utils.get_insert_key_dict(dii, (i, i), [])
        vdii.append(to)
        vdjj = utils.get_insert_key_dict(dii, (j, j), [])
        vdjj.append(to)

        vdji = utils.get_insert_key_dict(dij, (i, j), [])
        vdji.append(to)

        vdtt = utils.get_insert_key_dict(dtt, (t, t), [])
        vdtt.append(to)

        vdit = utils.get_insert_key_dict(dit, (i, t), [])
        vdit.append(to)
        vdjt = utils.get_insert_key_dict(dit, (j, t), [])
        vdjt.append(to)

    def dict2Array(d):
        d2 = dict()
        for k, v in d.items():
            v = np.asarray(v, dtype=int)
            d2[k] = v
        return d2

    return dict2Array(dii), dict2Array(dij), dict2Array(dit), dict2Array(dtt)


def genTrueNegTpl(adrId2Pairid, nDrug, nNegPerADR, kSpace=params.KSPACE):
    negTpls = []

    allPairs = set()
    for pairs in adrId2Pairid.values():
        for pair in pairs:
            allPairs.add(pair)

    for adrId, pairs in adrId2Pairid.items():
        adrId = adrId + nDrug
        ni = 0

        if kSpace:
            for pair in allPairs:
                d1, d2 = pair

                d1, d2 = swap(d1, d2)
                p = (d1, d2)
                if p not in pairs:
                    negTpls.append((d1, d2, adrId))
            continue

        nx = nNegPerADR * np.log(10) / (np.log(len(pairs) + 3))

        while ni < nx:
            d1, d2 = np.random.randint(0, nDrug, 2)
            d1, d2 = swap(d1, d2)
            pair = (d1, d2)
            if pair not in pairs:
                ni += 1
                negTpls.append((d1, d2, adrId))
    return negTpls


def genAdjacency(trainFold, numNode):
    adjacency = [[] for _ in range(numNode)]
    edgeSet = set()
    for triple in trainFold:
        d1, d2, se = triple
        d1, d2 = swap(d1, d2)
        for pair in [(d1, d2), (d1, se), (d2, se)]:
            if pair not in edgeSet:
                edgeSet.add(pair)
    for edge in edgeSet:
        u, v = edge
        adjacency[u].append(v)
        adjacency[v].append(u)
    return adjacency

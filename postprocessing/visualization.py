import numpy as np
import params
from postprocessing.plotFunc import *


def visual(realData, method, iFold):
    print("Visualize: ...")
    sortedADRs = realData.orderADRIds
    trainTpl = realData.trainTpl
    dataX = np.loadtxt("%s%sW_%s" % (params.EMBEDDING_PREX2, method, iFold))
    wx = np.loadtxt("%s%sW_Weight%s" % (params.EMBEDDING_PREX2, method, iFold))
    dADR2Name, dDrug2Name = utils.load_obj(params.ID2NamePath_TWOSIDE)
    print(len(dADR2Name), len(dDrug2Name))
    # dName2DrugId = utils.reverse_dict(dDrug2Name)
    # drugNamePairs = [[["Diazepam", "Clarithromycin"]],
    #                  [["Hydroxyzine", "Warfarin"]],
    #                  [["Simvastatin", "Glipizide"]],
    #                  [["Prednisone", "Tolterodine"]]]

    # DB:
    dd = {}
    for seId, pairs in realData.dADR2Drug.items():
        dd[dADR2Name[seId]] = len(pairs)
    kvs = utils.sort_dict(dd)
    print(kvs)

    print(sortedADRs[::-1][:10])
    print("Infrequent")
    for ii in sortedADRs[::-1][:10]:
        print(dADR2Name[ii], ii)
    nTop = 3
    seIds = sortedADRs[-nTop:]
    # seIds = sortedADRs[:nTop]
    print("Frequent")
    seNameList = []
    for ii in range(len(dADR2Name)):
        seNameList.append(dADR2Name[ii])

    allSeEmbedding = dataX[realData.nD:, :]
    allSeEmbedding = allSeEmbedding * wx
    print(allSeEmbedding.shape)
    print("Plotting all se")
    selectedSes = seIds
    for seId in seIds:
        print(dADR2Name[seId], seId)

    seName2Id = utils.reverse_dict(dADR2Name)
    names = ["panniculitis"]
    selectedSes = [seName2Id[name] for name in names]
    for se in selectedSes:
        plotData4(allSeEmbedding, offset=realData.nD, selectedSEs=[se], dADR2Name=dADR2Name)
    exit(-1)
    for ii in sortedADRs[:10]:
        print(dADR2Name[ii], ii)
    # drungIdPairs = []
    # for pList in drugNamePairs:
    #     vv = []
    #     for p in pList:
    #         d1, d2 = p
    #         vv.append([dName2DrugId[d1], dName2DrugId[d2]])
    #     drungIdPairs.append(vv)

    for ii, seId in enumerate(seIds):
        print(dADR2Name[seId], seId)
        w = wx[seId]
        dataXW = dataX * np.tile(w, (dataX.shape[0], 1))
        visual2(trainTpl, seId, realData.nD, dataXW, method, [], iFold, dADR2Name, dDrug2Name)
    exit(-1)
    return 0, 0, 0


def visual2(trainTpl, seId, nD, finalX, method, selectedPairs=[], iFold=5, dId2SeName={}, dId2DrugName={}):
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
    print("Num drug pairs", len(drugPairSet))
    mxPair = len(drugPairSet)
    drugIDList = list(drugIDSet)

    title = r'$\mathrm{CentSmoothie}$'
    plotData2(finalX, "%s_%s" % (method, iFold), title, offset=nD, sid=seIdOf, dPairs=drugPairSet[:mxPair],
              selectVDrugPair=selectedPairs, drugIDList=drugIDList, dSe2Name=dId2SeName,
              dDrug2Name=dId2DrugName, ndim=3
              )

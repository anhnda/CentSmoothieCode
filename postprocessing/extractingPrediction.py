import params
import numpy as np

from utils import utils


def loadTopPrediction(path):
    fin = open(path)
    dSe2Pairs = dict()
    currentSe = ""
    currentPairs = []
    while True:
        line = fin.readline()
        if line == "":
            break
        if line == "":
            continue
        if line.startswith("_"):
            continue
        if line[0].isalpha():
            currentSe = line.strip()
            currentPairs = []
            dSe2Pairs[currentSe] = currentPairs
            continue
        if line[0].startswith("\t"):
            triples = line.strip().split(",")
            # print(triples)
            currentPairs.append((triples[0].strip(), triples[1].strip(), float(triples[2])))
    return dSe2Pairs

def normalize(s):
    return s + 0.47
def checkTop():
    pathCent = "%s/Export_TOP_NEG_CentSmoothie_0" % params.OUTPUT_DIR
    pathHPNN = "%s/Export_TOP_NEG_HPNN_0" % params.OUTPUT_DIR
    pathDecagon = "%s/Export_TOP_NEG_Decagon_0" % params.OUTPUT_DIR

    dSe2PairCent = loadTopPrediction(pathCent)
    dSe2PairHPNN = loadTopPrediction(pathHPNN)
    dSe2PairDecagon = loadTopPrediction(pathDecagon)

    N_TOP_LIM = 200
    TOP = 10
    seList = ["panniculitis", "sarcoma", "pneumoconiosis", "splenectomy"]
    fout = open("%s/CheckTop.txt" % params.OUTPUT_DIR, "w")
    foutTriple = open("%s/Triple.txt" % params.OUTPUT_DIR, "w")
    fDrugList = open("%s/DrugList.txt" % params.OUTPUT_DIR, "w")
    drugList = set()

    def dInd(ll):
        d = {}
        for i, v in enumerate(ll):
            d1, d2, ss = v
            v = (d1, d2)
            d[v] = i + 1, ss
        return d

    fTT = open("%s/Latex.tex" % params.OUTPUT_DIR, "w")
    oldTop = [linex.strip().split(",") for linex in open("%s/oldtop.txt" % params.OUTPUT_DIR).readlines()]
    checkList = [linex.strip() for linex in open("%s/checkmark.txt" % params.OUTPUT_DIR).readlines()]
    ic = 0
    for ie, se in enumerate(seList):
        centPairs = dSe2PairCent[se][:TOP]
        dHPNN = dInd(dSe2PairHPNN[se][:N_TOP_LIM])
        dDecagon = dInd(dSe2PairDecagon[se][:N_TOP_LIM])
        fout.write("%s\n" % se)
        for ii, centPair in enumerate(centPairs):

            d1, d2, score = centPair
            centPair = (d1, d2)
            drugList.add(d1)
            drugList.add(d2)
            r1, score1 = utils.get_dict(dHPNN, centPair, (-1, 0))
            r2, score2 = utils.get_dict(dDecagon, centPair, (-1, 0))
            ll = checkList[ic]
            ic += 1
            if r2 == -1:
                lx = "-"
            else:
                lx = "%s(%.2f)" % (r2, score2)
            if ii == 0:
                d1, d2 = oldTop[ie * 3]


                fTT.write("\\multirow{3}{*}{%s} & %s, %s & %s(%.2f) & %s(%.2f) & %s & %s \\\\ \cline{2-6}\n" % (
                se.capitalize(), d1.strip(), d2.strip(), ii + 1, normalize(score), r1, score1, lx , ll))
            else:
                suff = "\\cline{2 - 6}"
                if ii == 9:
                    suff = "\\hline"
                if ii <= 2:
                    d1, d2 = oldTop[ie *3 + ii]
                fTT.write("& %s, %s & %s(%.2f) & %s(%.2f) & %s & %s \\\\ %s\n" % (d1.strip(), d2.strip(), ii + 1, normalize(score), r1, score1, lx, ll, suff))
            foutTriple.write("\"%s\", \"%s\", \"%s\"\n" % (d1, d2, se))
            fout.write("%s, %s, %s, %s\n" % ("%s, %s" % (d1, d2), ii + 1, r1, r2))
    fTT.close()
    fout.close()
    for drug in drugList:
        fDrugList.write("%s\n" % drug)
    fDrugList.close()
    foutTriple.close()


def checkLiterature():
    N_SE = 4
    TOP = 10
    fin = open("%s/CheckTop.txt" % params.OUTPUT_DIR)
    fOut = open("%s/CheckTopRef.txt" % params.OUTPUT_DIR, "w")

    fRef = open("")


if __name__ == "__main__":
    checkTop()

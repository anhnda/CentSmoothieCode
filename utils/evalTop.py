import numpy as np


def evalPreRec(top, predictedList, targetSet):
    prec = np.zeros(top, dtype=float)
    rec = np.zeros(top, dtype=float)
    t = min(top, len(predictedList))
    for i in range(t):
        v = predictedList[i]
        if v in targetSet:
            prec[i:] += 1
            rec[i:] += 1
    for i in range(top):
        prec[i] /= (i + 1)
        rec[i] /= len(targetSet)
    return prec, rec


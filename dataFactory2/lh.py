import numpy as np


def swap(d1, d2):
    if d1 > d2:
        d1, d2 = d2, d1
    return d1, d2


def getAFromL(L):
    d = np.diag(L)
    diag = np.diag(d)
    A = diag - L
    I = np.ones(L.shape[0])
    A = A + np.diag(I)
    A = A / np.max(A)

    return A, d


def getdegNormAfromL(L):
    d = np.diag(L)
    diag = np.diag(d)
    A = diag - L
    # A + I:
    I = np.ones(L.shape[0])
    A2 = A + np.diag(I)

    D = np.sum(A2, axis=1)
    D = np.power(np.sqrt(D), -1)
    DM12 = np.diag(D)
    normA = np.dot(np.dot(DM12, A2), DM12)

    return normA, d


def getAFromH(H):
    L = np.dot(H, H.transpose())
    return getAFromL(L)


def getUAFromEdges(edges, nSize):
    UH = []
    for edge in edges:
        d1, d2, se = edge
        ar = np.zeros(nSize)
        a1, a2 = swap(d1, d2)
        ar[a1] = 1
        ar[a2] = -1

        ar2 = np.zeros(nSize)
        ar2[se] = 1
        ar2[d1] = -1

        ar3 = np.zeros(nSize)
        ar3[se] = 1
        ar3[d2] = -1

        UH.append(ar)
        UH.append(ar2)
        UH.append(ar3)

    UH = np.vstack(UH).transpose()

    L = np.dot(UH, UH.transpose())
    return getAFromL(L)


def genLFromTpl(tpls, nSize):
    L = np.zeros((nSize, nSize), dtype=float)
    for tpl in tpls:
        d1, d2, si = tpl
        L[d1, d1] += 0.25
        L[d2, d2] += 0.25
        L[d1, d2] += 0.25
        L[d2, d1] += 0.25
        L[si, si] += 1

        L[d1, si] += -0.5
        L[si, d1] += -0.5
        L[d2, si] += -0.5
        L[si, d2] += -0.5
    return L


def genULFromTpl(tpls, nSize):
    L = np.zeros((nSize, nSize), dtype=float)

    def add(L, i, j):
        L[i, i] += 1
        L[j, j] += 1
        L[i, j] += -1
        L[j, i] += -1

    for tpl in tpls:
        d1, d2, si = tpl
        add(L, d1, d2)
        add(L, d1, si)
        add(L, d2, si)
    return L


def genAFromTpl(tpls, nSize):
    L = genLFromTpl(tpls, nSize)
    A, D = getAFromL(L)
    return A, D


def genUAFromTpl(tpls, nSize):
    UL = genULFromTpl(tpls, nSize)
    UA, UD = getAFromL(UL)
    return UA, UD


def genDegNormAFromTpl(tpls, nSize):
    L = genLFromTpl(tpls, nSize)
    A, D = getdegNormAfromL(L)

    return A, D


def genDegNormUAFromTpl(tpls, nSize):
    UL = genULFromTpl(tpls, nSize)
    UA, UD = getdegNormAfromL(UL)
    return UA, UD


def genDegANormFrom2Edges(edges, nSize):
    L = np.zeros((nSize, nSize))

    def add(L, i, j):
        L[i, i] += 1
        L[j, j] += 1
        L[i, j] += -1
        L[j, i] += -1

    for edge in edges:
        i, j = edge
        add(L, i, j)

    A, D = getdegNormAfromL(L)
    return A, D


def genAFrom2Edges(edges, nSize):
    L = np.zeros((nSize, nSize))

    def add(L, i, j):
        L[i, i] += 1
        L[j, j] += 1
        L[i, j] += -1
        L[j, i] += -1

    for edge in edges:
        i, j = edge
        add(L, i, j)

    A, D = getAFromL(L)
    return A, D

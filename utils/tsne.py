from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
def tsneDrugSe2(x,ofD, sz=1, iFold=0):
    labels = []

    for i in range(ofD):
        labels.append("D_%d" % (int(i/sz)))
    for i in range(ofD, x.shape[0]):
        labels.append("SSS_%d" % (i - ofD))

    d = TSNE(n_components=2).fit_transform(x)
    x = d[:, 0]
    y = d[:, 1]
    fig, ax = plt.subplots()

    ax.scatter(x, y)
    for i, label in enumerate(labels):
        ax.annotate(label, (x[i], y[i]))
    plt.title("IFold: %d" % iFold)
    return plt


def tsneDrugSe(x, ofD, g1, g2, s, sizeg):
    labels = []
    idsx = []
    for i in range(g1 * sizeg, (g1+1) * sizeg):
        labels.append("D_%s" % i)
        idsx.append(i)

    for i in range(g2 * sizeg, (g2+1) * sizeg):
        labels.append("D_%s" % i)
        idsx.append(i)

    labels.append("S_%d" % (s))

    idsx.append(s+ofD)
    idsx = np.asarray(idsx)
    print(idsx)
    sx = x[idsx]
    d = TSNE(n_components=2).fit_transform(sx)
    x = d[:, 0]
    y = d[:, 1]
    fig, ax = plt.subplots()

    ax.scatter(x, y)
    for i, label in enumerate(labels):
        ax.annotate(label, (x[i], y[i]))
    return plt

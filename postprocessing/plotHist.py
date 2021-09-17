import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


def plotHist(x, nBins, name, xlim=-1):
    fig, axs = plt.subplots()

    if xlim > 0:
        i = 0
        for i, xi in enumerate(x):
            if xi < xlim:
                break
        x = x[i:]
    axs.hist(x, bins=nBins)

    plt.xlabel("Num. Drug-drug pairs")
    plt.ylabel("Num. Side effects")

    plt.savefig("%s.png" % name)



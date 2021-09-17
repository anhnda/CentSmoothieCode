import pandas
import numpy as np
import config

MAX_HM = 2


def genPandasFromMatrix(mx):
    nRow, nCol = mx.shape
    rowIndices = []
    colNames = []
    for i in range(nRow):
        rowIndices.append("")
    for i in range(nCol):
        colNames.append("")
    df = pandas.DataFrame(mx, index=rowIndices, columns=colNames)
    # df = pandas.DataFrame(mx)
    return df


def plotMatrixHeatMap(mx, fileName=""):
    import seaborn as sns;
    import matplotlib

    import matplotlib.pyplot as plt
    # matplotlib.rcParams.update({'xtick.labelsize': 18})
    # matplotlib.rcParams.update({'ytick.labelsize': 18})
    # matplotlib.rcParams.update({'legend.fontsize': 20})

    data = genPandasFromMatrix(mx)

    f, axs = plt.subplots(figsize=(12, 3))
    # ax = sns.heatmap(data, ax=axs, cmap='plasma', vmin=0, vmax=MAX_HM, xticklabels=True, yticklabels=True)
    ax = sns.heatmap(data, ax=axs, cmap='plasma', xticklabels=True, yticklabels=True)

    # print ax
    # plt.show()
    plt.tight_layout()
    # plt.title("Distribution of %s side effects over 100 latent features"%nSiseEffect)
    if fileName != "":
        plt.savefig("%s/figs/%s.eps" % (config.C_DIR, fileName))
    # plt.show()

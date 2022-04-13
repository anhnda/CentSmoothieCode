import config
import re


def loadData():
    path = "%s/interRe" % config.OUTPUT_DIR
    modelNames = []
    precisions = []
    recalls = []
    fin = open(path)
    while True:
        line = fin.readline()
        if line == "":
            break

        modelName = line.strip()
        modelNames.append(modelName)

        line = fin.readline()
        line = re.sub("\s\s+", " ", line)
        parts = line.split(" ")
        prec = []
        for p in parts:

            prec.append(float(p))

        precisions.append(prec)

        line = fin.readline()
        line = re.sub("\s\s+", " ", line)
        parts = line.split(" ")
        recal = []
        for p in parts:
            recal.append(float(p))
        recalls.append(recal)
    TOP = [i for i in range(1, 11)]
    return modelNames, precisions, recalls, TOP


def plot():
    modelNames, precs, recals, X = loadData()
    import matplotlib.pyplot as plt

    TYPES = ['PRECISION', 'RECALL']
    RES = [precs, recals]
    LINE_STYLES = ['-', '--', ':']

    for j in range(2):
        typex = TYPES[j]
        rex = RES[j]
        fig = plt.figure()

        for i in range(len(modelNames)):
            label = modelNames[i]
            v = rex[i]
            plt.plot(X, v, label=label, linestyle=LINE_STYLES[i])
        plt.xlabel('TOP')
        plt.ylabel(typex)

        plt.legend(loc='lower right')
        plt.tight_layout()

        plt.savefig("%s/%s.eps" % (config.FIG_DIR, typex))
        plt.savefig("%s/%s.png" % (config.FIG_DIR, typex))


if __name__ == "__main__":
    plot()

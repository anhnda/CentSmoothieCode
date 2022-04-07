import numpy as np

import params


def getMeanSE(ar):
    mean = np.mean(ar)
    se = np.std(ar) / np.sqrt(len(ar))
    return mean, se


def getMeanSE2(ndar):
    mean = np.mean(ndar, axis=0)
    se = np.std(ndar, axis=0) / np.sqrt(ndar.shape[0])
    return mean, se


def export_from_log(path, indicator="@Iter  9900"):
    aucs, auprs = [], []
    acses = []
    fin = open(path)
    while True:
        line = fin.readline()
        if line == "":
            break
        if line.startswith(indicator):
            acse = fin.readline().strip()
            # print(acse)
            acses.append(eval(acse))
            aucaupr = fin.readline().strip()[:-1].split(",")
            auc, aupr = float(aucaupr[-2]), float(aucaupr[-1])
            aucs.append(auc)
            auprs.append(aupr)


    acses = np.asarray(acses)
    print(len(aucs), len(auprs), acses.shape)
    print("AUC: ", getMeanSE(aucs))
    print("AUPR: ", getMeanSE(auprs))
    print("AUC LIST: ", getMeanSE2(acses[:, :, 0]) )
    print("AUC LIST: ", getMeanSE2(acses[:, :, 1]) )


def run_export():
    path = "%s/CADDDI" % params.LOG_DIR
    export_from_log(path)


if __name__ == "__main__":
    run_export()

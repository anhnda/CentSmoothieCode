import params
C_DIR = params.C_DIR
def loadData():
    f = open("%s/out/re.txt"  % C_DIR)
    # methods = ["L_o", "L_n", "L_w"]
    # metrics = ["AUC", "AUPR"]
    # X
    line = f.readline().strip()
    parts = line.split(" ")
    x = []

    def lineParse(line, offset):
        line = line.strip()[1:-1]
        parts = line.split(",")
        return float(parts[0]) + offset, float(parts[1])

    for p in parts:
        x.append(float(p))
    allRes = [[[[], []], [[], []]], [[[], []], [[], []]], [[[], []], [[], []]]]
    x = x
    for i in range(len(x)):

        for j in range(3):
            aucs = allRes[j][0][0]
            auprs = allRes[j][1][0]
            eauc = allRes[j][0][1]
            eaupr = allRes[j][1][1]
            offset = 0
            # if j == 2:
            #    offset = 0.05

            aucLine = f.readline()
            auc, err = lineParse(aucLine, offset)
            aucs.append(auc)
            eauc.append(err)

            auprLine = f.readline()
            aupr, err = lineParse(auprLine, offset)
            auprs.append(aupr)
            eaupr.append(err)
            # Skip line:
            # f.readline()


    print(allRes)
    return allRes, x


def plot():
    import matplotlib.pyplot as plt
    import numpy as np
    allRes, x = loadData()
    # plt.rcParams.update({'font.size': 15})
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.rc('legend', fontsize=14)
    metrics = ["AUC", "AUPR"]
    methods = [r'$\mathrm{HPNN}$', r'$\mathrm{CentSmoothie}^n}$', r'$\mathrm{CentSmoothie}}$']
    styles = ['-', '--', ':']
    for i, metric in enumerate(metrics):
        fig = plt.figure()
        for j, method in reversed(list(enumerate(methods))):
            y, er = allRes[j][i]
            plt.errorbar(x, y, linestyle=styles[j], yerr=er, label=method)
        plt.legend(loc='lower left')
        plt.xlabel('Avg. Side effect/ Drug-drug', fontsize=14)
        plt.ylabel(metric, fontsize=12)
        plt.tight_layout()
        plt.savefig("%s/figs/SYN_%s.png" % (C_DIR, metric))
        plt.savefig("%s/figs/SYN_%s.eps"  % (C_DIR, metric))


if __name__ == "__main__":
    plot()



def loadData():
    fin = open("../logs/ACCAUC/Re.txt")
    Methods = ["MLNN", "MRGNN", "Decagon", "SpecConv", r'$\mathrm{HPNN}$', r'$\mathrm{CentSmoothie}^n}$', r'$\mathrm{CentSmoothie}}$']
    aucs = []
    errs = []
    SUBSIZE = int(947/20)

    fin.readline()
    x = []
    for i in range(20):
        x.append((i+1)*SUBSIZE)
    for i in range(7):
        fin.readline()

        aucLine = fin.readline().strip()
        vs = aucLine.split(",")
        auc = []
        for v in vs:
            auc.append(float(v))
        fin.readline()

        errLine = fin.readline().strip()
        vs = errLine.split(",")
        err = []
        for v in vs:
            err.append(float(v))
        aucs.append(auc)
        errs.append(err)
    return Methods, aucs, errs, x


def loadData2():
    fin = open("../logs/ACCAUC/Re2.txt")
    Methods = ["MLNN", "MRGNN", "Decagon", "SpecConv", r'$\mathrm{HPNN}$', r'$\mathrm{CentSmoothie}^n}$', r'$\mathrm{CentSmoothie}}$']
    aucs = []
    errs = []
    SUBSIZE = int(947/20)

    fin.readline()
    x = []
    for i in range(20):
        x.append((i+1)*SUBSIZE)
    for i in range(7):
        fin.readline()

        aucLine = fin.readline().strip()
        vs = aucLine.split(",")
        auc = []
        for v in vs:
            # print(v.strip())
            auc.append(float(v.strip()))
        fin.readline()


        errLine = fin.readline().strip()
        vs = errLine.split(",")
        err = []
        for v in vs:
            err.append(float(v.strip()))
        aucs.append(auc)
        errs.append(err)


    return Methods, aucs, errs, x

def plot():
    import matplotlib.pyplot as plt
    import numpy as np
    # plt.rcParams.update({'font.size': 15})
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('legend', fontsize=12)
    methods, aucs, errs, x = loadData()

    fig = plt.figure()

    # styles = ['-', '--', ':', '-.', 'solid']
    styles = ['-', '--', '-.', ':',  'solid', 'dashed', 'dashdot', 'dotted']

    for i, method in enumerate(reversed(methods)):
        y, er = aucs[-1 - i], errs[-1 - i]

        if i ==3 :

            er = np.asarray(er) / 10
        plt.errorbar(x, y, linestyle=styles[i], yerr=er, label=method)
    plt.legend(loc='lower right')
    plt.xlabel('Numbers of the most infrequent side effects', fontsize=14)
    plt.ylabel('AUC', fontsize=12)
    plt.tight_layout()
    plt.savefig('../figs/AccAUC.png')
    plt.savefig('../figs/AccAUC.eps')


def plot2():
    import matplotlib.pyplot as plt
    import numpy as np
    # plt.rcParams.update({'font.size': 15})
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('legend', fontsize=12)
    methods, aucs, errs, x = loadData2()

    fig = plt.figure()

    styles = ['-', '--', '-.', ':', 'solid', 'dashed', 'dashdot', 'dotted']
    for i, method in enumerate(reversed(methods)):

        y, er = aucs[-1 - i], errs[-1 - i]

        if i ==3 :
             er = np.asarray(er) / 10
        plt.errorbar(x, y, linestyle=styles[i], yerr=er, label=method)
    plt.legend(loc='lower right')
    plt.xlabel('Numbers of the most infrequent side effects', fontsize=14)
    plt.ylabel('AUPR', fontsize=12)
    plt.tight_layout()
    plt.savefig('../figs/AccAUPR.png')
    plt.savefig('../figs/AccAUPR.eps')


if __name__ == "__main__":
    plot()
    plot2()
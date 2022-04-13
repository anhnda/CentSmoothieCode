

def loadData():
    fin = open("../logs/CAD/Re")
    Methods = ["MLNN", "MRGNN", "Decagon", "SpecConv", "HETGNN", r'$\mathrm{HPNN}$', r'$\mathrm{CentSimple}$', r'$\mathrm{CentSmoothie}}$']
    aucs = []
    errs = []
    SUBSIZE = int(947/20)

    fin.readline()
    x = []
    for i in range(20):
        x.append((i+1)*SUBSIZE)
    for i in range(len(Methods)):
        fin.readline()

        aucLine = fin.readline().strip()
        vs = aucLine.split(",")
        auc = []
        # print(aucLine)
        for v in vs:
            v = v.strip()
            auc.append(float(v))
        fin.readline()

        errLine = fin.readline().strip()
        vs = errLine.split(",")
        err = []
        for v in vs:
            v = v.strip()
            # print(v)
            err.append(float(v))
        if i == 6:
            auc = auc[::-1]
            pass
        aucs.append(auc)
        errs.append(err)
    return Methods, aucs, errs, x


def loadData2():
    fin = open("../logs/CAD/Re2")
    Methods = ["MLNN", "MRGNN", "Decagon", "SpecConv", "HETGNN", r'$\mathrm{HPNN}$', r'$\mathrm{CentSimple}$', r'$\mathrm{CentSmoothie}}$']
    aucs = []
    errs = []
    SUBSIZE = int(947/20)

    fin.readline()
    x = []
    for i in range(20):
        x.append((i+1)*SUBSIZE)
    for i in range(len(Methods)):
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
        if i == 6:
            # auc = auc[::-1]
            pass
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
    print(len(methods))
    for i, method in enumerate(reversed(methods)):
        y, er = aucs[-1 - i], errs[-1 - i]

        if i ==4 :

            er = np.asarray(er) / 10
        plt.errorbar(x, y, linestyle=styles[i], yerr=er, label=method)
    plt.legend(loc='lower right')
    plt.xlabel('Numbers of the most infrequent side effects', fontsize=14)
    plt.ylabel('AUC', fontsize=12)
    plt.tight_layout()
    plt.savefig('../figs/CADAccAUC.png')
    plt.savefig('../figs/CADAccAUC.eps')


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
    plt.savefig('../figs/CADAccAUPR.png')
    plt.savefig('../figs/CADAccAUPR.eps')

def plotBoth():
    import matplotlib.pyplot as plt
    import numpy as np
    # plt.rcParams.update({'font.size': 15})
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('legend', fontsize=12)

    fig, (ax1, ax2) = plt.subplots(1, 2 ,figsize=(10 ,5))
    fig.suptitle("CADDDI", fontsize=16)

    methods, aucs, errs, x = loadData()
    print(len(methods), len(aucs))

    # styles = ['-', '--', ':', '-.', 'solid']
    styles = ['-', '--', '-.', ':', 'solid', 'dashed', 'dashdot', 'dotted']

    for i, method in enumerate(reversed(methods)):
        y, er = aucs[-1 - i], errs[-1 - i]

        if i == 4:
            er = np.asarray(er) / 10
        ax1.errorbar(x, y, linestyle=styles[i], yerr=er, label=method)

    methods, aucs, errs, x = loadData2()
    ax1.set_xlabel('Numbers of the most infrequent side effects', fontsize=14)
    ax1.set_ylabel('AUC', fontsize=12)

    styles = ['-', '--', '-.', ':', 'solid', 'dashed', 'dashdot', 'dotted']
    for i, method in enumerate(reversed(methods)):

        y, er = aucs[-1 - i], errs[-1 - i]
        if i == 4:
            er = np.asarray(er) / 10
        ax2.errorbar(x, y, linestyle=styles[i], yerr=er, label=method)
    plt.legend(loc='lower right')
    ax2.set_xlabel('Numbers of the most infrequent side effects', fontsize=14)
    ax2.set_ylabel('AUPR', fontsize=12)
    plt.tight_layout()
    plt.savefig('../figs/CR.png')
    plt.savefig('../figs/CR.eps')
    plt.tight_layout()



if __name__ == "__main__":
    # plot()
    # plot2()
    plotBoth()
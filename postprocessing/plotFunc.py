from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D

from matplotlib.text import Annotation

from utils import utils


class Annotation3D(Annotation):
    def __init__(self, text, xyz, *args, **kwargs):
        super().__init__(text, xy=(0, 0), *args, **kwargs)
        self._xyz = xyz

    def draw(self, renderer):
        x2, y2, z2 = proj_transform(*self._xyz, renderer.M)
        self.xy = (x2, y2)
        super().draw(renderer)


def _annotate3D(ax, text, xyz, *args, **kwargs):
    '''Add anotation `text` to an `Axes3d` instance.'''

    annotation = Annotation3D(text, xyz, *args, **kwargs)
    ax.add_artist(annotation)


setattr(Axes3D, 'annotate3D', _annotate3D)


def plotData(data, name, title, offset=-1, sid=-1, d1=-1, d2=-1, method="pca", ndim=3):
    fig = plt.figure()
    if method == "tnse":

        tsne = TSNE(n_components=ndim, verbose=1, perplexity=40, n_iter=300)
    else:
        tsne = PCA(n_components=ndim)

    tsne_results = tsne.fit_transform(data)

    if ndim == 2:
        df_subset = dict()
        df_subset['x'] = tsne_results[:, 0]
        df_subset['y'] = tsne_results[:, 1]
        sns.scatterplot(
            x="x", y="y",
            hue="y",
            # palette=sns.color_palette("hls", 10),
            data=df_subset,
            # legend="full",
            alpha=0.3
        )
        if offset > 0:
            nR, nC = data.shape
            for i in range(offset, nR):
                plt.annotate('S_%s' % (i - offset),
                             xy=(df_subset['x'][i], df_subset['y'][i]))
        plt.show()
        plt.savefig("figs/%s_%s_%s.png" % (name, method, ndim))

    else:
        x, y, z = tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2]
        rmx = max((np.max(np.fabs(x)), np.max(np.fabs(y)), np.max(np.fabs(z))))

        x = x / rmx
        y = y / rmx

        z = z / rmx
        ax = fig.add_subplot(111, projection='3d')
        nR, nC = data.shape
        color = 2 * np.pi * np.arange(0, nR, 1) / nR
        if offset > 0:
            mx = 10 * nR
            color1 = np.pi * np.arange(0, offset, 1) / mx
            ax.scatter3D(x[:offset], y[:offset], z[:offset], c='blue', marker='o', alpha=0.25, s=2)
            color2 = np.pi * np.arange(mx - (nR - offset), mx, 1) / mx
            ax.scatter3D(x[offset:], y[offset:], z[offset:], c='red', marker='^', alpha=0.25, s=2)

        else:
            ax.scatter3D(x, y, z, c=color, cmap='hsv', s=20)
        # if offset > 0:
        #      for i in range(offset, nR, 30):
        #          ax.text(x[i], y[i],z[i], 'S_%s'%(i-offset))

        seText = "S_%s" % (sid - offset)
        if sid - offset == 900:
            seText = "Sunburn"
        d1Text = 'D_%s' % (d1)
        d2Text = 'D_%s' % (d2)
        if d1 == 96:
            d1Text = "Prednisone"
        if d2 == 265:
            d2Text = "Leflunomide"
        # if sid > 0:
        #     ax.text(x[sid], y[sid], z[sid], seText)
        # if d1 > 0:
        #     ax.text(x[d1], y[d1], z[d1], d1Text)
        # if d2 > 0:
        #     ax.text(x[d2], y[d2], z[d2], d2Text)
        ax.scatter3D(x[sid], y[sid], z[sid], label='Side effect', c='red', marker='^', alpha=0.75, s=40)
        ax.scatter3D(x[d1], y[d1], z[d1], label='Drug', c='blue', marker='o', alpha=0.75, s=40)
        ax.scatter3D(x[d2], y[d2], z[d2], c='blue', marker='o', alpha=0.75, s=40)

        ax.annotate3D(seText, (x[sid], y[sid], z[sid]),
                      xytext=(30, -60),
                      textcoords='offset points',
                      bbox=dict(boxstyle="round", fc="salmon"),
                      arrowprops=dict(arrowstyle="-|>", ec='black', fc='white', lw=2))

        ax.annotate3D(d1Text, (x[d1], y[d1], z[d1]),
                      xytext=(30, 30),
                      textcoords='offset points',
                      bbox=dict(boxstyle="round", fc="skyblue"),
                      arrowprops=dict(arrowstyle="-|>", ec='black', fc='white', lw=2))
        p2 = (-60, 30)
        if name.__contains__("New"):
            print("in New")
            p2 = (10, 30)
        ax.annotate3D(d2Text, (x[d2], y[d2], z[d2]),
                      xytext=p2,
                      textcoords='offset points',
                      bbox=dict(boxstyle="round", fc="skyblue"),
                      arrowprops=dict(arrowstyle="-|>", ec='black', fc='white', lw=2), alpha=0.25)
        plt.title(title)
        # plt.show()
        plt.legend()
        plt.tight_layout()
        plt.savefig("figs/%s_%s_%s.png" % (name, method, ndim))
        plt.savefig("figs/%s_%s_%s.eps" % (name, method, ndim))


def plotData2(data, name, title, offset=-1, sid=-1, dPairs=[], selectVDrugPair = [], drugIDList=[], dSe2Name={}, dDrug2Name={}, method="pca",
              ndim=3, midpoint=True, nonre=False):
    fig = plt.figure()
    print(method, ndim, data.shape, offset, sid)
    if method == "tnse":

        dimReducer = TSNE(n_components=ndim, verbose=1, perplexity=40, n_iter=300)
    else:
        dimReducer = PCA(n_components=ndim)

    print(data.shape)

    noDrugList = []
    validDrugList = set(drugIDList)

    ll = [i for i in range(offset)]
    ll.append(sid)
    ll = np.asarray(ll)

    # ll2 = [i for i in range(offset)]

    # data = data[ll]
    # tsne_results = dimReducer.fit_transform(data)
    print(data.shape)
    tsne_results = dimReducer.fit_transform(data)
    tsne_results = tsne_results[ll]

    # if len(drugIDList) > 0:
    #     drugIDList.append(offset)
    #     data = data[drugIDList]
    #     dRemapDrug = dict()
    #     for i, v in enumerate(drugIDList):
    #         dRemapDrug[v] = i
    #     dRemapDrug[sid] = offset

    dRemapDrug = dict()
    for d in ll:
        dRemapDrug[d] = d
    dRemapDrug[sid] = offset

    # tsne_results = tsne_results[drugIDList]
    print(tsne_results.shape)
    #
    for i in range(offset):
        if i not in validDrugList:
            noDrugList.append(i)
    print("Start....")
    if ndim == 2:
        x, y = tsne_results[:, 0], tsne_results[:, 1]
        rmx = max((np.max(np.fabs(x)), np.max(np.fabs(y))))

        x = x / rmx
        y = y / rmx
        ax = fig.add_subplot(111)
        nR, nC = data.shape
        color = 2 * np.pi * np.arange(0, nR, 1) / nR
        if offset > 0:
            mx = 10 * nR
            color1 = np.pi * np.arange(0, offset, 1) / mx
            ax.scatter(x[:offset], y[:offset], c='blue', marker='o', alpha=0.01, s=2)
            color2 = np.pi * np.arange(mx - (nR - offset), mx, 1) / mx
            ax.scatter(x[offset:], y[offset:], c='red', marker='^', alpha=0.01, s=2)

        else:
            ax.scatter(x, y, c=color, cmap='hsv', s=20)
        # if offset > 0:
        #      for i in range(offset, nR, 30):
        #          ax.text(x[i], y[i],z[i], 'S_%s'%(i-offset))

        seText = utils.get_dict(dSe2Name, sid - offset, "S_%s" % (sid - offset))
        # seText = "%s_%s" % (seText, sid -offset)
        seText = seText.capitalize()
        rsid = dRemapDrug[sid]
        if len(dPairs) > 0:
            for ii, p in enumerate(dPairs):
                d1, d2 = p
                rd1, rd2 = dRemapDrug[d1], dRemapDrug[d2]
                d1Text = utils.get_dict({}, d1, 'D_%s' % (d1))
                d2Text = utils.get_dict({}, d2, 'D_%s' % (d2))
                d1Text = "%s_%s" % (d1Text, ii)
                d2Text = "%s_%s" % (d2Text, ii)
                # if sid > 0:
                #     ax.text(x[sid], y[sid], z[sid], seText)
                # if d1 > 0:
                #     ax.text(x[d1], y[d1], z[d1], d1Text)
                # if d2 > 0:
                #     ax.text(x[d2], y[d2], z[d2], d2Text)
                if ii == 0:
                    ax.scatter(x[rsid], y[rsid], label='Side effect', c='red', marker='^', alpha=0.75, s=40)
                    ax.scatter(x[rd1], y[rd1], label='Drug', c='blue', marker='o', alpha=0.75, s=3)
                    if midpoint:
                        ax.scatter((x[rd1] + x[rd2]) / 2, (y[rd1] + y[rd2]) / 2,
                                     label='Midpoint', marker='*', c='black', s=1)

                else:
                    ax.scatter(x[rsid], y[rsid], c='red', marker='^', alpha=0.75, s=3)
                    ax.scatter(x[rd1], y[rd1], c='blue', marker='o', alpha=0.75, s=3)
                ax.scatter(x[rd2], y[rd2], c='blue', marker='o', alpha=0.75, s=3)

                ax.plot([x[rd1], x[rd2]], [y[rd1], y[rd2]], c='gray', alpha=0.5, linewidth='1')
                if midpoint:
                    ax.scatter((x[rd1] + x[rd2]) / 2, (y[rd1] + y[rd2]) / 2,  c='black',
                                 marker='*', s=1)
            ax.annotate(seText, (x[rsid], y[rsid]),
                          xytext=(30, 60),
                          textcoords='offset points',
                          bbox=dict(boxstyle="round", fc="salmon"),
                          arrowprops=dict(arrowstyle="-|>", ec='black', fc='white', lw=2))
        plt.title(title)
        # plt.show()
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig("figs/%s_%s_%s_%s.png" % (name, method, ndim, seText))
        # plt.savefig("figs/%s_%s_%s_%s.eps" % (name, method, ndim, seText))

    else:
        print("AAAAAAAAAAAAAAAAAA 1")
        dLable = 'Drug'
        if nonre:
            dLable = 'Relevant drug'
        x, y, z = tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2]
        rmx = max((np.max(np.fabs(x)), np.max(np.fabs(y)), np.max(np.fabs(z))))

        x = x / rmx
        y = y / rmx
        z = z / rmx
        ax = fig.add_subplot(111, projection='3d')
        nR, nC = data.shape
        color = 2 * np.pi * np.arange(0, nR, 1) / nR
        if offset > 0:
            mx = 10 * nR
            color1 = np.pi * np.arange(0, offset, 1) / mx
            ax.scatter3D(x[:offset], y[:offset], z[:offset], c='blue', marker='o', alpha=0.01, s=2)
            color2 = np.pi * np.arange(mx - (nR - offset), mx, 1) / mx
            ax.scatter3D(x[offset:], y[offset:], z[offset:], c='red', marker='^', alpha=0.01, s=2)

        else:
            ax.scatter3D(x, y, z, c=color, cmap='hsv', s=20)
        # if offset > 0:
        #      for i in range(offset, nR, 30):
        #          ax.text(x[i], y[i],z[i], 'S_%s'%(i-offset))

        seText = utils.get_dict(dSe2Name, sid - offset, "S_%s" % (sid - offset))
        # seText = "%s_%s" % (seText, sid -offset)
        seText = seText.capitalize()
        rsid = dRemapDrug[sid]
        if len(dPairs) > 0:
            for ii, p in enumerate(dPairs):
                d1, d2 = p
                rd1, rd2 = dRemapDrug[d1], dRemapDrug[d2]
                d1Text = utils.get_dict({}, d1, 'D_%s' % (d1))
                d2Text = utils.get_dict({}, d2, 'D_%s' % (d2))
                d1Text = "%s_%s" % (d1Text, ii)
                d2Text = "%s_%s" % (d2Text, ii)
                # if sid > 0:
                #     ax.text(x[sid], y[sid], z[sid], seText)
                # if d1 > 0:
                #     ax.text(x[d1], y[d1], z[d1], d1Text)
                # if d2 > 0:
                #     ax.text(x[d2], y[d2], z[d2], d2Text)
                if ii == 0:
                    ax.scatter3D(x[rsid], y[rsid], z[rsid], label='Side effect', c='red', marker='^', alpha=0.75, s=40)
                    ax.scatter3D(x[rd1], y[rd1], z[rd1], label=dLable, c='blue', marker='o', alpha=0.75, s=3)
                    if midpoint:
                        ax.scatter3D((x[rd1] + x[rd2]) / 2, (y[rd1] + y[rd2]) / 2, (z[rd1] + z[rd2]) / 2,
                                     label='Midpoint', marker='*', c='black', s=1)

                else:
                    ax.scatter3D(x[rsid], y[rsid], z[rsid], c='red', marker='^', alpha=0.75, s=3)
                    ax.scatter3D(x[rd1], y[rd1], z[rd1], c='blue', marker='o', alpha=0.75, s=3)
                ax.scatter3D(x[rd2], y[rd2], z[rd2], c='blue', marker='o', alpha=0.75, s=3)

                ax.plot([x[rd1], x[rd2]], [y[rd1], y[rd2]], [z[rd1], z[rd2]], c='gray', alpha=0.5, linewidth='1')
                if midpoint:
                    ax.scatter3D((x[rd1] + x[rd2]) / 2, (y[rd1] + y[rd2]) / 2, (z[rd1] + z[rd2]) / 2, c='black',
                                 marker='*', s=1)
            ax.annotate3D(seText, (x[rsid], y[rsid], z[rsid]),
                          xytext=(30, 60),
                          textcoords='offset points',
                          bbox=dict(boxstyle="round", fc="salmon"),
                          arrowprops=dict(arrowstyle="-|>", ec='black', fc='white', lw=2))

            if len(selectVDrugPair) > 0:
                for p in selectVDrugPair:
                    print(p)
                    for vdi in p:
                        print(vdi)
                        vdText = dDrug2Name[vdi]
                        xvdi, yvdi, zvdi = tsne_results[vdi,0], tsne_results[vdi,1], tsne_results[vdi,2]
                        ax.annotate3D(vdText, (xvdi, yvdi, zvdi),
                                       xytext=(30, 30),
                                       textcoords='offset points',
                                       bbox=dict(boxstyle="round", fc="skyblue"),
                                       arrowprops=dict(arrowstyle="-|>", ec='black', fc='white', lw=2))



            if nonre:
                for ii, d in enumerate(noDrugList):
                    if ii == 0:
                        ax.scatter3D(x[d], y[d], z[d], label='Non-relevant drug', c='green', marker='.', alpha=0.5, s=3)
                    ax.scatter3D(x[d], y[d], z[d], c='green', marker='.', alpha=0.75, s=3)

        plt.title(title)
        # plt.show()
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig("figs/%s_%s_%s_%s.png" % (name, method, ndim, seText))
        plt.savefig("figs/%s_%s_%s_%s.eps" % (name, method, ndim, seText))
        # plt.savefig("figs/%s_%s_%s_%s.pdf" % (name, method, ndim, seText))





def plotData3(data, title="", offset=-1, selectedSEs = [], dADR2Name = {}, method="tnse",
              ndim=3):
    fig = plt.figure()
    print(method, ndim, data.shape, offset)
    if method == "tnse":

        dimReducer = TSNE(n_components=ndim, verbose=1, perplexity=40, n_iter=300)
    else:
        dimReducer = PCA(n_components=ndim)

    print(data.shape)


    tsne_results = dimReducer.fit_transform(data)

    # tsne_results = dimReducer.fit_transform(data)
    # tsne_results = tsne_results[ll]

    # if len(drugIDList) > 0:
    #     drugIDList.append(offset)
    #     data = data[drugIDList]
    #     dRemapDrug = dict()
    #     for i, v in enumerate(drugIDList):
    #         dRemapDrug[v] = i
    #     dRemapDrug[sid] = offset

    print("AAAAAAAAAAAAAAAAAA 1")

    x, y, z = tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2]
    rmx = max((np.max(np.fabs(x)), np.max(np.fabs(y)), np.max(np.fabs(z))))

    x = x / rmx
    y = y / rmx
    z = z / rmx
    ax = fig.add_subplot(111, projection='3d')
    nR, nC = data.shape
    color = 2 * np.pi * np.arange(0, nR, 1) / nR

    ax.scatter3D(x, y, z, c='blue', marker='o', alpha=0.01, s=2)
    for seId in selectedSEs:
        # seId -= offset
        seText = utils.get_dict(dADR2Name, seId, "S_%s" % seId).capitalize()
        ax.annotate3D(seText, (x[seId], y[seId], z[seId]),
                      xytext=(30, 60),
                      textcoords='offset points',
                      bbox=dict(boxstyle="round", fc="salmon"),
                      arrowprops=dict(arrowstyle="-|>", ec='black', fc='white', lw=2))

    plt.title(title)
    # plt.show()
    # plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig("figs/AllSe.eps")
    # plt.savefig("figs/%s_%s_%s_%s.pdf" % (name, method, ndim, seText))




def plotData4(data, title="", offset=-1, selectedSEs = [], dADR2Name = {}, method="tnse",
              ndim=3):
    fig = plt.figure()
    print(method, ndim, data.shape, offset)
    if method == "tnse":

        dimReducer = TSNE(n_components=ndim, verbose=1, perplexity=40, n_iter=300)
    else:
        dimReducer = PCA(n_components=ndim)

    print(data.shape)


    tsne_results = dimReducer.fit_transform(data)

    # tsne_results = dimReducer.fit_transform(data)
    # tsne_results = tsne_results[ll]

    # if len(drugIDList) > 0:
    #     drugIDList.append(offset)
    #     data = data[drugIDList]
    #     dRemapDrug = dict()
    #     for i, v in enumerate(drugIDList):
    #         dRemapDrug[v] = i
    #     dRemapDrug[sid] = offset

    print("AAAAAAAAAAAAAAAAAA 1")

    x, y, z = tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2]
    rmx = max((np.max(np.fabs(x)), np.max(np.fabs(y)), np.max(np.fabs(z))))

    x = x / rmx
    y = y / rmx
    z = z / rmx
    ax = fig.add_subplot(111, projection='3d')
    nR, nC = data.shape
    color = 2 * np.pi * np.arange(0, nR, 1) / nR

    ax.scatter3D(x, y, z, c='blue', marker='.', alpha=0.01, s=2)

    from scipy.spatial import distance_matrix as dm
    mm = dm(tsne_results, tsne_results)
    np.fill_diagonal(mm, 10000)

    mainSe = ""
    for seId in selectedSEs:
        # seId -= offset
        ax.scatter3D(x[seId], y[seId], z[seId], c='red', marker='o', alpha=1, s=3)

        seText = utils.get_dict(dADR2Name, seId, "S_%s" % seId).capitalize()
        if mainSe == "":
            mainSe = seText
        ax.annotate3D(seText, (x[seId], y[seId], z[seId]),
                      xytext=(30, 60),
                      textcoords='offset points',
                      bbox=dict(boxstyle="round", fc="salmon"),
                      arrowprops=dict(arrowstyle="-|>", ec='red', fc='white', lw=2))


        ids = np.argsort(mm[seId, :])
        vv = [ids[0], ids[1]]
        si = vv[0]
        seText = utils.get_dict(dADR2Name, si, "S_%s" % si).capitalize()
        print(seText)
        ax.scatter3D(x[si], y[si], z[si], c='red', marker='o', alpha=1, s=3)

        ax.annotate3D(seText, (x[si], y[si], z[si]),
                      xytext=(-90, 30),
                      textcoords='offset points',
                      bbox=dict(boxstyle="round", fc="salmon"),
                      arrowprops=dict(arrowstyle="-|>", ec='black', fc='white', lw=2), zorder=2)

        si = vv[1]
        seText = utils.get_dict(dADR2Name, si, "S_%s" % si).capitalize()
        print(seText)
        ax.scatter3D(x[si], y[si], z[si], c='red', marker='o', alpha=1, s=3)

        ax.annotate3D(seText, (x[si], y[si], z[si]),
                      xytext=(-90, -55),
                      textcoords='offset points',
                      bbox=dict(boxstyle="round", fc="salmon"),
                      arrowprops=dict(arrowstyle="-|>", ec='black', fc='white', lw=2), zorder=2)

    plt.title(title)
    # plt.show()
    # plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig("figs/Rel_%s.eps" % mainSe)
    # plt.savefig("figs/%s_%s_%s_%s.pdf" % (name, method, ndim, seText))

from utils import utils

from utils.logger.logger2 import MyLogger
from models.trainCentSmoothie import WrapperWeightCentSmooth
from models.hegnn.trainHEGNN import WrapperHEGNN
from models.hpnn.trainHPNN import WrapperHPNN
from models.decagon.trainDecagon import WrapperDecagon
from dataFactory.dataLoader import DataLoader
import params

import numpy as np
import random
import torch


class Runner:
    def __init__(self, model="CentSmoothie"):
        resetRandomSeed()

        self.data = None

        utils.ensure_dir("%s/logs" % params.C_DIR)
        if model.upper().startswith("CENT"):
            self.wrapper = WrapperWeightCentSmooth()
        elif model.upper().startswith("HPNN"):
            self.wrapper = WrapperHPNN()
        elif model.upper().startswith("DECA"):
            self.wrapper = WrapperDecagon()
        elif model.upper().startswith("HE"):
            self.wrapper = WrapperHEGNN()
        else:
            print("Error: Unknown model:", model)
            exit(-1)

        PREX = "RL_%s_%s" % (params.MAX_R_ADR, params.MAX_R_DRUG)
        logPath = "%s/logs/%s_%s_%s_%s" % (
        params.C_DIR, PREX, self.wrapper.name, params.L_METHOD, utils.getCurrentTimeString())
        self.logger = MyLogger(logPath)
        self.wrapper.setLogger(self.logger)

    def run(self):
        method = params.L_METHOD
        aucs = []
        auprs = []
        aucks = []
        auprks = []
        self.logger.infoAll(("Laplacian: ", method, " Deg Norm: ", params.DEG_NORM, "N_LAYER: ", params.N_LAYER))
        self.logger.infoAll(
            ("On Weight: ", params.ON_W, "Learn Weight: ", params.LEARN_WEIGHT_IN, params.LEARN_WEIGHT_LAST))
        self.logger.infoAll(("On Layer W: ", params.LAYER_WEIGHT))
        self.logger.infoAll(("ON REAL: ", params.ON_REAL))
        self.logger.infoAll(("Embedding size: ", params.EMBEDDING_SIZE))
        self.logger.infoAll(("FORCE CPU: ", params.FORCE_CPU))
        self.logger.infoAll(("Visual:", params.VISUAL))
        self.logger.infoAll(("N_SGD, LW, LR", params.N_SGD, params.L_W, params.LAMBDA_R))
        ar = [i for i in range(params.K_FOLD)]

        ss = ar
        # ss = [i for i in range(10)]

        ss = [0]
        print(ss)

        for iFold in ss:
            resetRandomSeed()
            self.logger.infoAll(("Fold: ", iFold))
            wrapper = DataLoader()
            wrapper.loadData(iFold, dataPref=params.D_PREF)
            self.logger.infoAll(("NDRUG, NSE: ", wrapper.data.nD, wrapper.data.nSe))

            auc, aupr, vv = self.wrapper.train(wrapper, iFold, method)
            if params.VISUAL:
                continue
            aucs.append(auc)
            auprs.append(aupr)
            if vv != -1:
                vv = np.asarray(vv)
                aucks.append(vv[:, 0])
                auprks.append(vv[:, 1])
            self.logger.infoAll(("AUC, AUPR: ", auc, aupr))

        mauc, eauc = getMeanSE(aucs)
        maupr, eaupr = getMeanSE(auprs)

        self.logger.infoAll(params.L_METHOD)
        self.logger.infoAll((mauc, eauc))
        self.logger.infoAll((maupr, eaupr))

        if len(aucks) > 0:
            aucks = np.vstack(aucks)
            auprks = np.vstack(auprks)
            self.logger.infoAll((getMeanSE2(aucks)))
            self.logger.infoAll((getMeanSE2(auprks)))




def getMeanSE(ar):
    mean = np.mean(ar)
    se = np.std(ar) / np.sqrt(len(ar))
    return mean, se


def getMeanSE2(ndar):
    mean = np.mean(ndar, axis=0)
    se = np.std(ndar, axis=0) / np.sqrt(ndar.shape[0])
    return mean, se


def convertArToString(ar):
    s = ""
    for v in ar:
        s += "," + str(v)
    return "[" + s[1:] + "]"


def resetRandomSeed():
    random.seed(params.TORCH_SEED)
    torch.manual_seed(params.TORCH_SEED)
    np.random.seed(params.TORCH_SEED)

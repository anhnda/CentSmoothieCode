import numpy as np
from utils import utils
import params
import torch


class Wrapper:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pass

    def loadData(self, iFold):
        self.iFold = iFold
        print("Loading iFold: ", iFold)
        folder = params.FOLD_DATA

        data = utils.load_obj("%s/_%d_%d_%d_%d" % (
            folder, params.MAX_R_ADR, params.MAX_R_DRUG, params.ADR_OFFSET, iFold))

        ddiTensor = np.zeros((data.nD, data.nD, data.nSe))


        train2Label = data.pTrainPair2Label
        test2Label = data.pTestPosLabel
        valid2Label = data.pValidPosLabel
        negTest2Label = data.pTestNegLabel
        indices = []
        for edge, label in train2Label.items():
            d1, d2 = edge
            for l in label:
                indices.append((d1, d2, l))
                indices.append((d2, d1, l))
                # print(d1, d2, l)
        ddiTensor[tuple(np.transpose(indices))] = 1

        testPosIndices = []
        validPosIndices = []
        testNegIndices = []

        for edge, label in test2Label.items():
            d1, d2 = edge
            for l in label:
                testPosIndices.append((d1, d2, l))
                testPosIndices.append((d2, d1, l))

        testPosIndices = tuple(np.transpose(testPosIndices))

        for edge, label in valid2Label.items():
            d1, d2 = edge
            for l in label:
                validPosIndices.append((d1, d2, l))
                validPosIndices.append((d2, d1, l))

        validPosIndices = tuple(np.transpose(validPosIndices))
        for edge, label in negTest2Label.items():
            d1, d2 = edge
            for l in label:
                testNegIndices.append((d1, d2, l))
                testNegIndices.append((d2, d1, l))



        testNegIndices = tuple(np.transpose(testNegIndices))


        self.ddiTensor = ddiTensor
        features = data.drug2Features
        if not params.PROTEIN_FEATURE:
            features = data.drug2Features[:, :881]
        self.features = torch.from_numpy(features).float().to(self.device)

        self.x = torch.from_numpy(ddiTensor).float().to(self.device)

        self.testNegIndices = testNegIndices
        self.validPosIndices  = validPosIndices
        self.testPosIndices = testPosIndices
        self.data = data




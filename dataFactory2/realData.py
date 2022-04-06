class RealData:
    def __init__(self, trainFolds, testFolds, validFolds, AFolds, UAFolds, negFold, features):
        self.trainFolds = trainFolds
        self.testFolds = testFolds
        self.validFolds = validFolds
        self.AFolds = AFolds
        self.UAFolds = UAFolds
        self.negFold = negFold
        self.drug2Features = features

        self.featureSize = self.drug2Features.shape[1]


class RealFoldData:
    def __init__(self, trainFold, testFold, validFold, AFold, UAFold, negFold, features):
        self.trainFold = trainFold
        self.testFold = testFold
        self.validFold = validFold
        self.AFold = AFold
        self.UAFold = UAFold
        self.negFold = negFold
        self.drug2Features = features

        self.featureSize = self.drug2Features.shape[1]

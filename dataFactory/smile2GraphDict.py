from pysmiles import read_smiles
import params
from utils import utils

class SMILEGraph():
    def loadDrugName2Info(self):
        f = open(params.DRUGBANK_ATC_INCHI)
        dDrugName2Inchi = dict()
        dDrugName2SMILE = dict()
        while True:
            line = f.readline()
            if line == "":
                break
            parts = line.strip().split("\t")
            dDrugName2Inchi[parts[1].lower()] = parts[-1]
            dDrugName2SMILE[parts[1].lower()] = parts[4]
        f.close()
        self.drugName2Inchi = dDrugName2Inchi
        self.drugInchi2Name = utils.reverse_dict(dDrugName2Inchi)
        self.drugName2SMILE = dDrugName2SMILE
        return dDrugName2Inchi


    def saveSmile2Graph(self):
        self.loadDrugName2Info()
        dSmile2Graph = dict()
        for smile in self.drugName2SMILE.values():
            graph = read_smiles(smile)
            dSmile2Graph[smile] = graph

        utils.save_obj(dSmile2Graph, params.SMILE2GRAPH)

if __name__ == "__main__":
    smileGraph = SMILEGraph()
    smileGraph.saveSmile2Graph()

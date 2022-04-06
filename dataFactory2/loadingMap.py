import params
from utils.utils import loadMapSetFromFile, loadMapFromFile


def loadDrugProteinMap():
    from utils.utils import loadMapSetFromFile

    def mergeDict(d1, d2):
        # Remove "" element
        from utils import utils
        d = {}
        for k1, v1 in d1.items():
            vm = set()
            v2 = utils.get_dict(d2, k1, set())
            for vi in (v1, v2):
                for vs in vi:
                    if vs != "":
                        vm.add(vs)
            d[k1] = vm
        for k2, v2 in d2.items():
            if k2 not in d1:
                vm = set()
                for vs in v2:
                    if vs != "":
                        vm.add(vs)
                d[k2] = vm
        return d

    # Load BindingDB:
    dDrugProteinBindingDB = loadMapSetFromFile(params.BINDINGDB_EXTENDED)
    # print "BindingDB", len(dDrugProteinBindingDB)

    # Load Drugbank data:
    dDrugProteinDrugBank = loadMapSetFromFile(params.DRUGBANK_INCHI_PROTEIN, "|", sepValue=",")
    # print "DrugBank", len(dDrugProteinDrugBank)

    dDrugProtein = mergeDict(dDrugProteinBindingDB, dDrugProteinDrugBank)

    return dDrugProtein


def loadProtein2Pathway():
    dProtein2Pathway = loadMapSetFromFile(params.KEGG_UNIPROT_PATHWAY_MAPPING_PATH, sep="|", sepValue=",")
    return dProtein2Pathway

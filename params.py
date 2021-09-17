import os
C_DIR = os.path.abspath(os.path.dirname(__file__))

DATA_DIR = "%s/data" % C_DIR
OUTPUT_DIR = "%s/out" % C_DIR
FIG_DIR = "%s/figs" % C_DIR


DRUGBANK_ATC_INCHI = "%s/DrugBank/ATC_2_Inchikeys.txt" % DATA_DIR
PPI_FILE = "%s/HPRD/PPI_UNIPROT.txt" % DATA_DIR
PUBCHEM_FILE = "%s/PubChem/Inchikey2ProfileFilled.dat" % DATA_DIR
POLY_ADR_FILE  = "%s/PolyADR/polyDrugADR.txt" % DATA_DIR
MONO_ADR_FILE = "%s/PolyADR/mergeredDrugAdr.txt" % DATA_DIR
DUMP_POLY = "%s/PolyADR/dump.dat" % DATA_DIR
R_DATA = "%s/PolyADR/realData.dat" % DATA_DIR
ID2NamePath = "%s/Id2Name" % OUTPUT_DIR
EMBEDDING_PREX = "%s/Embeding_" % OUTPUT_DIR
BINDINGDB_EXTENDED = "%s/DrugBank/BindingDTB" % DATA_DIR
DRUGBANK_INCHI_PROTEIN = "%s/DrugBank/DRUBBANK_INCHIKEY_2_PROTEIN.dat" % DATA_DIR
FOLD_DATA = "%s/PolyADR/Folds" % DATA_DIR
KEGG_UNIPROT_PATHWAY_MAPPING_PATH = "%s/KEGG/uniprot_2_pathway.txt" % DATA_DIR
SMILE2GRAPH = "%s/DrugBank/SMILDE2GRAPH.dat" % DATA_DIR
OPTIMIZER = "Adam"



FORCE_SKIP_OPTIONS = False
FORCE_CPU = False



K_FOLD = 20

N_ITER = 3000
ITER_DB = 100


TORCH_SEED = int('1100110011010100111011011010011', 2)



PROTEIN_FEATURE = False

PRINT_DB = True





MAX_R_ADR = 1000
MAX_R_DRUG = 600
ADR_OFFSET = 0
SAMPLE_NEG = 800


L_METHOD = "Cent"

ON_REAL = True
DEG_NORM = True


ON_W = True
LEARN_WEIGHT_IN = True
LEARN_WEIGHT_LAST = True

N_SGD = 20

N_LAYER = 2
L_W = 0.8
LAMBDA_R = 0.1
R_TYPE = "L1"


LAYER_WEIGHT = True
EMBEDDING_SIZE = 20


NUM_LISTI_WORKER = 10
N_DATA_WORKER = 5

N_SEC = 20
S_KFOLD = 5
VISUAL = False
EXPORT_TOP_NEG = False



from optparse import OptionParser
from dataFactory.genData import genData
from utils import utils


def convertStringToBoolean(val):
    print(val)
    if type(val) == bool:
        return val
    if val.upper() == "TRUE":
        return True
    elif val.upper() == "FALSE":
        return False
    else:
        print("Fatal error: Wrong Boolean String")
        exit(-1)


def parseConfig(options):
    print(options)
    params.FORCE_CPU = options.oncpu
    params.N_ITER = options.iter
    params.N_LAYER = options.layer
    params.EMBEDDING_SIZE = options.emb
    params.VISUAL = options.visual
    params.DEG_NORM = True
    params.ON_REAL = True
    params.ON_W = True
    params.D_PREF = options.data.upper()
    params.EXPORT_TOP_NEG = options.export

    if params.D_PREF == "S":
        params.DEG_NORM = False
        params.ON_REAL = False
        params.P_SYN = options.psyn
        params.N_SEC = 1

    if options.simple:
        params.LEARN_WEIGHT_LAST = False
        params.LEARN_WEIGHT_IN = False

    if options.model.upper().startswith("HP"):
        options.model = "HPNN"
    elif options.model.upper().startswith("CENT"):
        options.model = "CentSmoothie"
    elif options.model.upper().startswith("DECA"):
        options.model = "Decagon"
    elif options.model.upper().startswith("HE"):
        options.model = "HEGNN"

    params.MODEL = options.model

def checkDir():
    utils.ensure_dir(params.TMP_DIR)
    utils.ensure_dir(params.LOG_DIR)
    utils.ensure_dir(params.OUTPUT_DIR)
    utils.ensure_dir(params.FIG_DIR)


def runMode(opts):
    from models.runner import Runner
    print("Run model...")
    runner = Runner(model=opts.model)
    runner.run()


def visual(opts):
    from dataFactory.dataLoader import DataLoader
    from postprocessing.visualization import visual
    wrapper = DataLoader()
    wrapper.loadData(opts.fold, dataPref=params.D_PREF)
    visual(wrapper.data, opts.model, opts.fold)


def export(opts):

    from dataFactory.dataLoader import DataLoader
    from postprocessing.exportTopNeg import exportTopNeg
    wrapper = DataLoader()
    wrapper.loadData(opts.fold, dataPref=params.D_PREF)
    exportTopNeg(wrapper.data, opts.model, opts.fold)

if __name__ == "__main__":
    import params

    checkDir()

    parser = OptionParser()

    parser.add_option("-c", "--oncpu", dest="oncpu", action="store_true")
    parser.add_option("-r", "--run", dest="run", action="store_true")
    parser.add_option("-x", "--export", dest="export", action="store_true")
    parser.add_option("-f", "--fold", dest="fold", type='int', default=0)

    parser.add_option("-m", "--model", dest="model", type='str', default="CentSmoothie")
    parser.add_option("-s", "--simple", dest="simple", action="store_true")
    parser.add_option("-v", "--visual", dest="visual", action="store_true")
    parser.add_option("-i", "--iter", dest="iter", type='int', default=params.N_ITER)
    parser.add_option("-y", "--layer", dest="layer", type='int', default=params.N_LAYER)
    parser.add_option("-e", "--emb", dest="emb", type='int', default=params.EMBEDDING_SIZE)
    parser.add_option("-g", "--gen", dest="gen", action="store_true")
    parser.add_option("-p", "--psyn", dest="psyn", type=float, default=0.5)
    parser.add_option("-d", "--data", dest="data", type='str', default="",
                      help="data prefix, either '' for TWOSIDES, 'C' for CADDDI, or 'C'for JADERDDI")

    (options, args) = parser.parse_args()

    parseConfig(options)

    if options.gen:
        from models.runner import resetRandomSeed

        resetRandomSeed()
        print("Generating data...")
        genData.genDataByPref(options.data)
        exit(-1)

    elif options.visual:
        print("Visualization...")
        visual(options)
        exit(-1)
    elif options.export:
        print("Exporting...")
        export(options)
        exit(-1)
    elif options.run:
        print("Training...")
        runMode(options)

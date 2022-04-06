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
    params.L_METHOD = "Cent"
    params.N_ITER = options.iter
    params.N_LAYER = options.layer
    params.EMBEDDING_SIZE = options.emb
    params.VISUAL = options.visual
    params.DEG_NORM = True
    params.ON_REAL = True
    params.ON_W = True
    params.D_PREF = options.data


def checkDir():
    utils.ensure_dir(params.TMP_DIR)
    utils.ensure_dir(params.LOG_DIR)


def runMode():
    from models.runner import Runner
    print("Run model...")
    runner = Runner()
    runner.run()


if __name__ == "__main__":
    import params

    checkDir()

    parser = OptionParser()

    parser.add_option("-c", "--oncpu", dest="oncpu", action="store_true")
    parser.add_option("-r", "--run", dest="run", action="store_true")
    parser.add_option("-v", "--visual", dest="visual", action="store_true")
    parser.add_option("-i", "--iter", dest="iter", type='int', default=params.N_ITER)
    parser.add_option("-y", "--layer", dest="layer", type='int', default=params.N_LAYER)
    parser.add_option("-e", "--emb", dest="emb", type='int', default=params.EMBEDDING_SIZE)
    parser.add_option("-g", "--gen", dest="gen", action="store_true")
    parser.add_option("-d", "--data", dest="data", type='str', default="",
                      help="data prefix, either '' for TWOSIDES, 'C' for CADDDI, or 'C'for JADERDDI")

    (options, args) = parser.parse_args()

    parseConfig(options)

    if options.gen:
        from models.runner import resetRandomSeed

        resetRandomSeed()

        if options.gen:
            print("Generating data...")
            genData.genDataByPref(options.data)
            exit(-1)

        exit(-1)
    if options.run:
        runMode()

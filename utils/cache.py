import config
import os
from sklearn.externals import joblib
C_DIR = os.path.abspath(os.path.dirname(__file__))
CACHE_DIR = ".__caches"
FULL_CACHE_DIR = "%s/../%s"%(C_DIR,CACHE_DIR)

def saveCache(obj,name):
    joblib.dump(obj,"%s/%s"%(FULL_CACHE_DIR,name))

def loadCache(name):
    fullPath = "%s/%s"%(FULL_CACHE_DIR,name)
    try:
        obj = joblib.load(fullPath)
    except:
        obj = -1
    return obj


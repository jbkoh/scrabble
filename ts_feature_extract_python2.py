
# coding: utf-8

# In[2]:

# Basic Modules
import numpy as np
import pandas as pd
from scipy.cluster.vq import *
import operator
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt   
import shelve
import re
from collections import Counter, defaultdict, OrderedDict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics#v_measure_score
import scipy
from sklearn.feature_extraction import DictVectorizer
from matplotlib.backends.backend_pdf import PdfPages
import csv
import sys
import math
from copy import deepcopy
import random
from datetime import datetime, timedelta
import pickle
import sys
from multiprocessing import Pool
import pp
import shelve
from collections import OrderedDict
import json

# ML modules
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn import preprocessing
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import BaggingClassifier
#from sklearn.svm import OneClassSVM
#from sklearn.mixture import GMM
#from sklearn.mixture import DPGMM
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
#from sklearn.neural_network import BernoulliRBM as RBM


# In[3]:

with open('metadata/bacnet_devices.json','r') as fp:
    sensor_dict = json.load(fp)


# In[4]:

#building_name = 'EBU3B'
building_name = 'AP_M'
outputFilename = 'model/fe_' + building_name.lower() + '.json'

with open('metadata/%s_sentence_dict.json'%building_name.lower(), 'r') as fp:
    sentence_dict = json.load(fp)
srcidList = sentence_dict.keys()


# In[5]:

with open('metadata/building_info.json', 'r') as fp:
    building_dict = json.load(fp)
nae_list = building_dict[building_name]


# In[6]:

def extract_features(srcidList, dummy):
    import feature_extractor as fe
    import pandas as pd
    resultList = list()
    invalidSrcidList = list()
    for srcid in srcidList:
        try:
            filename = 'data/'+srcid+'.csv'
            ts = pd.Series.from_csv(filename, header=0)
            resultList.append((srcid, fe.get_features(ts)))
        except:
            invalidSrcidList.append(srcid)
            continue
    print invalidSrcidList
    return resultList


# In[ ]:

#extract_features(srcidList,None)


# In[9]:

#p = Pool(4)
#tempDict = dict((p.map(extract_features, srcidList)))

ppservers = ()
ncpus = 4
rangeList = list()

#srcidList = ['505_0_3000043', '506_0_3000026', '505_0_3000003', '506_0_3000023',  '506_0_3000027']

sensorsNum = len(srcidList)
for i in range(0,ncpus):
    rangeList.append(range(sensorsNum/ncpus*(i+1) - sensorsNum/ncpus, sensorsNum/ncpus*(i+1)))
print "=============="

jobServer = pp.Server(ncpus, ppservers=ppservers)
print "Starting pp with", jobServer.get_ncpus(), "workers"
jobList = list()
for oneRange in rangeList:
    #print [srcidList[i] for i in oneRange]
    jobList.append(jobServer.submit(extract_features, ([srcidList[i] for i in oneRange], True)))

resultList = list()
resultList = [0,0,0,0]
for i, job in enumerate(jobList):
    resultList[i] = job()
#r1 = jobList[0]()
#r2 = jobList[1]()
jobServer.wait()
jobServer.print_stats()
print "=-============"
#print r1
#print r2

dictList = list()
#print resultList
for result in resultList:
#    print "result: ", result
    dictList = dictList + result
resultDict = dict(dictList)
print "Done"

#job_server = pp.Server(ncpus, ppservers=ppservers)
#j1 = job_server.submit(extract_features, srcidList[0:1500])
with open(outputFilename, 'w') as fp:
    json.dump(resultDict, fp)

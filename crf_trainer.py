
# coding: utf-8

# In[1]:

from functools import reduce
import pycrfsuite
import json
import pandas as pd
import numpy as np
#import brick_parser
#reload(brick_parser)
#from brick_parser import tagList, tagsetList, equipTagsetList, pointTagsetList, locationTagsetList,equalDict, pointTagList, equipTagList, locationTagList, equipPointDict
import random


# In[ ]:

buildingName = 'ebu3b'
with open('metadata/%s_label_dict.json'%buildingName, 'r') as fp:
    labelListDict = json.load(fp)
with open("metadata/%s_sentence_dict.json"%buildingName, "r") as fp:
    sentenceDict = json.load(fp)


# In[ ]:

adder = lambda x,y:x+y
totalWordSet = set(reduce(adder, sentenceDict.values(), []))


# In[ ]:

#def calc_features(sentence,labels):
def calc_features(sentence):
    sentenceFeatures = list()
    for i, word in enumerate(sentence):
        features = [
            'word.lower=' + word.lower(),
            'word.isdigit=%s' % word.isdigit()
        ]
        if i>0:
            features.extend([
                    '-1:word.lower=' + sentence[i-1].lower()
                ])
        else:
            features.append('BOS')
            
        if i<len(sentence)-1:
            features.extend([
                    '+1:word.lower=' + sentence[i+1].lower(),
                ])
        else:
            features.append('EOS')
        sentenceFeatures.append(features)
    return sentenceFeatures


# In[ ]:

#get_ipython().run_cell_magic(u'time', u'', u'trainer = pycrfsuite.Trainer(verbose=False)\n#for srcid, setence in sentenceDict.items():\n\nrandomIdxList = random.sample(range(0,60),60)\nfor i, (srcid, labels) in enumerate(labelListDict.items()):\n    if i not in randomIdxList:\n        continue\n    sentence = sentenceDict[srcid]\n    #trainer.append(pycrfsuite.ItemSequence(calc_features(sentence, labels)), labels)\n    trainer.append(pycrfsuite.ItemSequence(calc_features(sentence)), labels)')
trainer = pycrfsuite.Trainer(verbose=False)
#for srcid, setence in sentenceDict.items():

randomIdxList = random.sample(range(0,len(labelListDict)),len(labelListDict))
for i, (srcid, labels) in enumerate(labelListDict.items()):
    if i not in randomIdxList:
        continue
    sentence = sentenceDict[srcid]
    trainer.append(pycrfsuite.ItemSequence(calc_features(sentence)), labels)


# In[ ]:

trainer.train('random.crfsuite')


# In[ ]:

tagger = pycrfsuite.Tagger()
tagger.open('random.crfsuite')


# In[ ]:

predictedDict = dict()
for srcid, sentence in sentenceDict.items():
    predicted = tagger.tag(calc_features(sentence))
    predictedDict[srcid] = predicted

# In[ ]:

#srcid = '505_14_3001723'
precisionOfTrainingDataset = 0.0
totalWordCount = 0.0

randIdxList = random.sample(range(0,4000), 20)
for srcid in labelListDict.keys():
#for i, srcid in enumerate(sentenceDict.keys()):
    #if not i in randIdxList:
    #    continue
    print("===================== %s ========================== "%srcid)
    sentence = sentenceDict[srcid]
    predicted = predictedDict[srcid]
    if not srcid in labelListDict.keys():
        for word, predTag in zip(sentence, predicted):
            print('{:20s} {:20s}'.format(word,predTag))
    else:
        printing_pairs = list()
        for word, predTag, origLabel in zip(sentence, predicted, labelListDict[srcid]):
            printing_pair = [word,predTag,origLabel]
            if origLabel!='none':
                if predTag==origLabel:
                    precisionOfTrainingDataset += 1
                    printing_pair = ['O'] + printing_pair
                else:
                    printing_pair = ['X'] + printing_pair
                    #print("WRONG BEGIN")
                    #print('{:20s} {:20s} {:20s}'.format(word,predTag,origLabel))
                    #print("WRONG END")
                totalWordCount += 1
            else:
                printing_pair = ['O'] + printing_pair
            printing_pairs.append(printing_pair)
        if 'X' in [pair[0] for pair in printing_pairs]:
            for (flag, word, predTag, origLabel) in printing_pairs:
                print('{:5s} {:20s} {:20s} {:20s}'\
                        .format(flag, word, predTag, origLabel))


    print("===============================================")


# In[ ]:

print(precisionOfTrainingDataset/totalWordCount)


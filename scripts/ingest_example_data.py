import pdb
import json

import pandas as pd

import sys
sys.path.insert(0, '/home/jbkoh/repo/scrabble_public')
from scrabble.data_model import *

point_postfixes = ['sensor',
                   'setpoint',
                   'status',
                   'alarm',
                   'command',
                   ]

def choose_point_tagset(tagsets):
    # This should be more accurate with using the schema
    point_tagset = ''
    for tagset in tagsets:
        postfix = tagset.split('_')[-1].lower()
        if postfix in point_postfixes:
            point_tagset = tagset
            break
    assert point_tagset, 'Point TagSet is not found in {0}'.format(tagsets)
    return point_tagset



building = 'example'

# Store RawMetadata
df = pd.read_csv('metadata/example_rawmetadata.csv', index_col='SourceIdentifier')
for srcid, row in df.iterrows():
    srcid = str(srcid)
    doc = RawMetadata\
        .objects(srcid=srcid, building=building)\
        .upsert_one(srcid=srcid, building=building)
    doc.metadata = row.to_dict()
    doc.save()

# Store LabeledMetadata
fullparsinsgs = json.load(open('metadata/example_char_labels.json', 'r'))
tagsets = json.load(open('metadata/example_tagsets.json', 'r'))
for srcid, fullparsing in fullparsinsgs.items():
    doc = LabeledMetadata\
        .objects(srcid=srcid, building=building)\
        .upsert_one(srcid=srcid, building=building)
    doc.fullparsing = fullparsing
    doc.tagsets = tagsets[srcid]
    doc.point_tagset = choose_point_tagset(doc.tagsets)
    doc.save()

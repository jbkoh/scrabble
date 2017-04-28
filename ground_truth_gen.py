import pdb
import json

import pandas as pd

from brick_parser import equipTagsetList as equip_tagsets, \
                        locationTagsetList as location_tagsets

building = 'ap_m'

sensor_df = pd.read_csv('metadata/{0}_sensor_types_location.csv'\
                        .format(building)).set_index('Unique Identifier')

with open('metadata/{0}_label_dict_justseparate.json'\
            .format(building), 'r') as fp:
    label_dict = json.load(fp)
with open('metadata/{0}_sentence_dict_justseparate.json'\
            .format(building), 'r') as fp:
    sentence_dict = json.load(fp)

nonpoint_tagsets = equip_tagsets + location_tagsets + ['networkadapter']

def find_nonpoint_tagsets(tagset):
    if tagset.split('-')[0] in nonpoint_tagsets:
        return tagset
    else:
        return ''

truth_dict = dict()
for srcid, label_list in label_dict.items():
    sentence = sentence_dict[srcid]
    phrase_list = list()
    truth_list = list()
    sentence_meanings = [(token,label) 
                         for token, label 
                         in zip(sentence, label_list) 
                         if label not in ['none', 'unknown']]
    right_identifier_buffer = ''
    for (token, label) in sentence_meanings:
        if label=='leftidentifier':
#            phrase_list[-1] += ('-' + token)
            continue
        elif label=='rightidentifier':
#            right_identifier_buffer += token
            continue

        phrase_list.append(label)
        if right_identifier_buffer:
            phrase_list[-1] += ('-' + right_identifier_buffer)
    truth_list = [phrase 
                  for phrase 
                  in phrase_list 
                  if find_nonpoint_tagsets(phrase)]
    truth_list.append(sensor_df['Schema Label'][srcid].replace(' ', '_'))
    truth_dict[srcid] = list(set(truth_list))

    # TODO: add all labels to a dict (except point type info)

with open('metadata/{0}_ground_truth.json'.format(building), 'w') as fp:
    json.dump(truth_dict, fp, indent=2)

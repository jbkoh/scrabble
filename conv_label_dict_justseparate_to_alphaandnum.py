import json
import re

building_name = 'ebu3b'

with open('metadata/%s_label_dict.json' % building_name, 'r') as fp:
    label_dict = json.load(fp)
with open('metadata/%s_sentence_dict_justseparate.json' % building_name, 'r') as fp:
    sentence_dict = json.load(fp)

new_label_dict = dict()
for srcid, label_list in label_dict.items():
    new_label_list = list()
    sentence = sentence_dict[srcid]
    for word, label in zip(sentence, label_list):
        if len(re.findall("[a-zA-Z]+|\d+", word))>0:
            new_label_list.append(label)
    new_label_dict[srcid] = new_label_list

with open('metadata/ebu3b_label_dict_alphaandnum.json', 'w') as fp:
    json.dump(new_label_dict, fp, indent=4, separators=(',', ': '))

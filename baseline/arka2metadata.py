import json
import re
import pdb

divider_dict_dict = {
        'AlphaAndNum':
        {                                                                   
        'smk_alm': {                                                               
            ('smk', 'alm'): ('smoke', 'alarm')                                     
            },                                                                     
        'low_rat': {                                                               
            ('low', 'rat'): ('low', 'return air temperature')                      
            },                                                                     
        'i__vr': {                                                                 
            ('i', 'vr'): ('integrate', 'variable')                                  
            },                                                                     
        'vav_av': {                                                                
            ('vav', 'av'): ('vav', 'average discharge air pressure sensor')         
            },                                                                     
        'cnd_flow': {                                                              
            ('cnd', 'flow'): ('condenser', 'water flow variable')                  
            },                                                                     
        'f_flow': {                                                                
            ('f', 'flow'): ('filtered', 'flow sensor')                             
            },                                                                     
        'vav_mx': {                                                                
            ('vav', 'mx'): ('vav', 'max discharge air pressure sensor')            
            }
        },
        'JustSeparate':{
        'smk_alm': {                                                               
            ('smk', '_', 'alm'): ('smoke', 'alarm')                                     
            },                                                                     
        'low_rat': {                                                               
            ('low', '_', 'rat'): ('low', 'return air temperature')                      
            },                                                                     
        'i__vr': {                                                                 
            ('i', '__', 'vr'): ('integrate', 'variable')                                  
            },                                                                     
        'vav_av': {                                                                
            ('vav','_', 'av'): ('vav', 'average discharge air pressure sensor')         
            },                                                                     
        'cnd_flow': {                                                              
            ('cnd', '_', 'flow'): ('condenser', 'water flow variable')                  
            },                                                                     
        'f_flow': {                                                                
            ('f', '_', 'flow'): ('filtered', 'flow sensor')                             
            },                                                                     
        'vav_mx': {                                                                
            ('vav', '_', 'mx'): ('vav', 'max discharge air pressure sensor')            
            }
        }
        }

itemmer = lambda x:x.items()
adder = lambda x,y:x+y

#divider_dict = dict(reduce(adder, map(itemmer, divider_dict.values()), []))

known_label_dict = {
            'soda': 'building ahu',
            'sodc': 'building chilled/condensor water loop',
            'sodh': 'building hot water loop',
            'sods': 'building supply fan',
            'smk': 'smoke',
            'alm': 'alarm',
            'low': 'low',
            'rat': 'return air temperature',
            'i': 'integrate',
            'vr': 'variable',
            'vav': 'vav',
            'av': 'average discharge air pressure',
            'cnd': 'condenser',
            'flow': 'flow',
            'f': 'filtered',
            'mx': 'max discharge air pressure sensor'
}

sentence_dict = dict()
label_dict = dict()

tokenization_rule_dict = {
        'AlphaAndNum': "[a-zA-Z]+|\d+",
        'JustSeparate':"([a-zA-Z]+|\d+|[^0-9a-z])"
        }

for token_type in ['AlphaAndNum', 'JustSeparate']:
    with open('Berkeley_SodaHall_MetadataParseKey', 'r') as fp:
        for srcid, line in enumerate(fp):
            line = line.replace(" ", "").replace("\t", "").replace("\n", "")
            raw_metadata, line = line.split("=>")
            sentence = re.findall(tokenization_rule_dict[token_type],
                    raw_metadata.lower())
            sentence_dict[srcid] = sentence

            item_list = line.split(";")
            item_dict = dict()
            ref_dict = dict()
            for item in item_list:
                if item=="":
                    continue
                key, val = item.split("->")
           #     if "Ref" in val:
           #         ref_dict[val] = key
           #     else:
                item_dict[key.lower()] = val.lower()

            item_dict = dict(known_label_dict.items() + item_dict.items())
            label_list = [None for _ in sentence]

            
            for i, word in enumerate(sentence):
                if label_list[i]:
                    continue
                if word in item_dict.keys():
                    label_list[i] = item_dict[word]
                else:
                        label_list[i] = 'none'
            label_dict[srcid] = label_list

    with open("soda_sentence_dict_%s.json" % token_type.lower(), 'w') as fp:
        json.dump(sentence_dict, fp, indent=4, separators=(',', ': '))

    with open('soda_label_dict_%s.json' % token_type.lower(), 'w') as fp:
        json.dump(label_dict, fp, indent=4, separators=(',', ': '))

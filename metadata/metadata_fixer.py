import json
import pdb


building_name = 'ap_m'
with open('{0}_sentence_dict_justseparate.json'.format(building_name), 'r') as fp:
    sentence_dict = json.load(fp)
with open('{0}_label_dict_justseparate.json.backup'.format(building_name), 'r') as fp:
#with open('{0}_label_dict_justseparate.json'.format(building_name), 'r') as fp:
    label_dict = json.load(fp)

new_label_dict = dict()
fault_cnt = 0
for i, (srcid, label_list) in enumerate(label_dict.items()):
    try:
        sentence = sentence_dict[srcid]
        new_label_list = list()
        label_list.reverse()
        for word in sentence:
            if word==' ':
                new_label_list.append('none')
            else:
                new_label_list.append(label_list.pop())
        assert(len(new_label_list)==len(sentence))
        assert(len(label_list)==0)
        new_label_dict[srcid] = new_label_list
    except:
        fault_cnt += 1

print(fault_cnt)

with open('{0}_new_label_dict.json'.format(building_name), 'w') as fp:
    json.dump(new_label_dict, fp, indent=2)

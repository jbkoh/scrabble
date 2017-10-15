import json
from common import *
from collections import Counter
from functools import reduce
import pdb

source_buildings = ['ap_m', 'bml']
target_building = 'bml'

with open('result/crf_entity_iter_{0}{1}_char2tagset_iter_0.json'
              .format( ''.join(source_buildings), target_building), 'r') as fp:
    data = json.load(fp)


c_dict = get_cluster_dict(target_building)
c_counters = []
check_in = lambda x,y: x in y
srcid_nums = []
c_sizes = []
for datum in data:
    learning_srcids = set(datum['learning_srcids'])
    cids = reduce(adder, [find_keys(srcid, c_dict, crit=check_in) 
                          for srcid in learning_srcids])
    c_counters.append(Counter(cids))
    srcid_nums.append(len(learning_srcids) - 200)
    c_sizes.append(len(c_counters[-1]))
pdb.set_trace()
    

import json
from collections import OrderedDict
import random
import pdb

def select_random_samples(cluster_filename, srcids, n, use_cluster_flag, token_type='alphaandnum'):  

    with open(cluster_filename, 'r') as fp:                                     
        cluster_dict = json.load(fp)                                            
                                                                                
    # Learning Sample Selection                                                 
    sample_srcids = set()                                                       
    length_counter = lambda x: len(x[1])                                        
    ander = lambda x, y: x and y                                                
    if use_cluster_flag:                                                        
        sample_cnt = 0                                                          
        sorted_cluster_dict = OrderedDict(                                      
            sorted(cluster_dict.items(), key=length_counter, reverse=True))         
        while len(sample_srcids) < n:                                           
            for cluster_num, srcid_list in sorted_cluster_dict.items():  
                # pdb.set_trace()       
                valid_srcid_list = set(srcid_list).intersection(set(srcids)).difference(set(sample_srcids))                         
                if len(valid_srcid_list) > 0:                                   
                    sample_srcids.add(random.choice(list(valid_srcid_list)))              
                if len(sample_srcids) >= n:                                     
                    break                                                       
    else:                                                                       
#        random_idx_list = random.sample(range(0,len(srcids)),n)                            
#        sample_srcids = [labeled_srcid_list[i] for i in random_idx_list]           
        sample_srcids = random.sample(srcids, n)                                
    return list(sample_srcids)

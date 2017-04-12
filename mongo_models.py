from copy import copy


summary_query_template = {
        'source_building': None,
        'target_building': None,
        'source_sample_num': None,
        'label_type': None,
        'token_type': None,
        'use_cluster_flag': None
        }
        

def get_summary_query_template():
    return copy(summary_query_template)

from copy import copy

from pymongo import MongoClient

c =  MongoClient()
db = c.get_database('scrabble')


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


model_template = {
        'source_list': [{'building_name':[]}],
        'model_binary': 'bson.binary.Binary',
        'gen_time': 'datetime',
        'use_cluster_flag': bool,
        'token_type': 'justseparate',
        'label_type': 'label'
        }

result_template = {
        # Metadata
        'label_type': '',
        'token_type': '',
        'use_cluster_flag': '',
        'source_cnt_list': [['building1', 100], ['building2', 10]],
        'target_building': 'building3',
        # Result
        }

def get_model(query):
    docs = db.get_collection('model').find(query)
    if not query.get('gen_time'):
        docs = docs.sort('gen_time',-1).limit(1)
    return docs[0]

def store_model(model):
    db.get_collection('model').insert_one(model)

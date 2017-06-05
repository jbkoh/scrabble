from copy import copy
import pdb
import code

from pymongo import MongoClient

C = MongoClient()
DB = C.get_database('scrabble')


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

def build_query(q):
    if not q.get('label_type'):
        label_type = 'label'
    else:
        label_type = q['label_type']
    if not q.get('use_cluster_flag'):
        use_cluster_flag = True
    else:
        use_cluster_flag = q['use_cluster_flag']
    building_list = q['building_list']
    query = {
        '$and':[{
            'label_type': label_type,
            'use_cluster_flag': use_cluster_flag,
            'source_building_count': len(building_list),
        }]
    }
    query['source_cnt_list'] = []
    query['target_building'] = q['target_building']
    for building, source_sample_num in \
            zip(building_list, q['source_sample_num_list']):
        query['$and'].append(
            {'source_list.{0}'.format(building): {'$exists': True}})
        query['$and'].append({'$where': \
                                    'this.source_list.{0}.length={1}'.\
                                    format(building, source_sample_num)})
        query['source_cnt_list'].append([building, source_sample_num])
    query['$and'].append({'source_building_count':len(building_list)})
    if q.get('learning_srcids'):
        query['$and'].append({'learning_srcids': q['learning_srcids']})
    return query

def get_model(query):
    docs = DB.get_collection('model').find(query)
    if not query.get('gen_time'):
        docs = docs.sort('gen_time', -1)#.limit(1)
    #print('Using the model generated at {0}'.format(docs[0]['date']))
    print('Using the model generated at {0}'.format(docs[0]['gen_time']))
    return docs[0]

def store_model(model):
    DB.get_collection('model').insert_one(model)

def store_result(results):
    DB.get_collection('results').insert_one(results)

def get_entity_results(query):
    docs = DB.get_collection('results').find(query)
    if not query.get('gen_time'):
        docs = docs.sort('gen_time', -1).limit(1)
    if docs.count()>0:
        doc = docs[0]
        return doc
    else:
        return None

def get_tags_mapping(query):
    docs = DB.get_collection('results').find(query)
    if not query.get('gen_time'):
        docs = docs.sort('gen_time', -1).limit(1)
    doc = docs[0]
    return doc['result']

def get_crf_results(query, n=1):
    if query.get('learning_srcids'):
        normalized_query = {'learning_srcids': sorted(query['learning_srcids'])}
    else:
        normalized_query = build_query(query)
    docs = DB.get_collection('results').find(normalized_query)
    if not query.get('date'):
        docs = docs.sort('date', -1)
    if docs.count()>0:
        docs = docs.limit(n)
        print('Using the model generated at {0}'.format(docs[0]['date']))
        return docs[0]
    else:
        return None

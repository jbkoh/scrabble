import argparse

from common import *
from char2ir import crf_test, learn_crf_model
from ir2tagsets import entity_recognition_from_ground_truth_get_avg, entity_recognition_iteration

def str2bool(v):
    if v in ['true', 'True']:
        return True
    elif v in ['false', 'False']:
        return False
    else:
        assert(False)

def str2slist(s):
    s.replace(' ', '')
    return s.split(',')

def str2ilist(s):
    s.replace(' ', '')
    return [int(c) for c in s.split(',')]

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.register('type','bool',str2bool)
    parser.register('type','slist', str2slist)
    parser.register('type','ilist', str2ilist)

    parser.add_argument(choices=['learn', 'predict', 'entity', 'crf_entity', \
                                 'init', 'result'],
                        dest = 'prog')

    parser.add_argument('predict',
                         action='store_true',
                         default=False)

    """
    parser.add_argument('-b',
                        type=str,
                        help='Learning source building name',
                        dest='source_building')
    parser.add_argument('-n', 
                        type=int, 
                        help='The number of learning sample',
                        dest='sample_num')
    """

    parser.add_argument('-bl',
                        type='slist',
                        help='Learning source building name list',
                        dest='source_building_list')
    parser.add_argument('-nl',
                        type='ilist',
                        help='A list of the number of learning sample',
                        dest='sample_num_list')
    parser.add_argument('-l',
                        type=str,
                        help='Label type (either label or category',
                        default='label',
                        dest='label_type')
    parser.add_argument('-c',
                        type='bool',
                        help='flag to indicate use hierarchical cluster \
                                to select learning samples.',
                        default=False,
                        dest='use_cluster_flag')
    parser.add_argument('-d',
                        type='bool',
                        help='Debug mode flag',
                        default=False,
                        dest='debug_flag')
    parser.add_argument('-t',
                        type=str,
                        help='Target buildling name',
                        dest='target_building')
    parser.add_argument('-eda',
                        type='bool',
                        help='Flag to use Easy Domain Adapatation',
                        default=False,
                        dest='eda_flag')
    parser.add_argument('-ub',
                        type='bool',
                        help='Use Brick when learning',
                        default=False,
                        dest='use_brick_flag')
    parser.add_argument('-avg',
                        type=int,
                        help='Number of exp to get avg. If 1, ran once',
                        dest='avgnum',
                        default=1)
    parser.add_argument('-iter',
                        type=int,
                        help='Number of iteration for the given work',
                        dest='iter_num',
                        default=1)
    parser.add_argument('-wk',
                        type=int,
                        help='Number of workers for high level MP',
                        dest='worker_num',
                        default=2)
    parser.add_argument('-nj',
                        type=int,
                        help='Number of processes for multiprocessing',
                        dest='n_jobs',
                        default=4)
    parser.add_argument('-ct',
                        type=str,
                        help='Tagset classifier type. one of RandomForest, \
                              StructuredCC.',
                        dest='tagset_classifier_type',
                        default='StructuredCC')
    parser.add_argument('-ts',
                        type='bool',
                        help='Flag to use time series features too',
                        dest='ts_flag',
                        default=False)
    parser.add_argument('-neg',
                        type='bool',
                        help='Negative Samples augmentation',
                        dest='negative_flag',
                        default=True)
    parser.add_argument('-exp', 
                        type=str,
                        help='type of experiments for result output',
                        dest = 'exp_type')
    parser.add_argument('-post', 
                        type=str,
                        help='postfix of result filename',
                        default='0',
                        dest = 'postfix')

    args = parser.parse_args()

    tagset_classifier_type = args.tagset_classifier_type

    if args.prog == 'learn':
        learn_crf_model(building_list=args.source_building_list,
                        source_sample_num_list=args.sample_num_list,
                        token_type='justseparate',
                        label_type=args.label_type,
                        use_cluster_flag=args.use_cluster_flag,
                        debug_flag=args.debug_flag,
                        use_brick_flag=args.use_brick_flag)
    elif args.prog == 'predict':
        crf_test(building_list=args.source_building_list,
                 source_sample_num_list=args.sample_num_list,
                 target_building=args.target_building,
                 token_type='justseparate',
                 label_type=args.label_type,
                 use_cluster_flag=args.use_cluster_flag,
                 use_brick_flag=args.use_brick_flag)
    elif args.prog == 'entity':
        if args.avgnum == 1:
            entity_recognition_iteration(args.iter_num,
                                         args.source_building_list,
                                         args.sample_num_list,
                                         args.target_building,
                                         'justseparate',
                                         args.label_type,
                                         args.use_cluster_flag,
                                         args.use_brick_flag,
                                         args.debug_flag,
                                         args.eda_flag,
                                         args.ts_flag,
                                         args.negative_flag,
                                         args.n_jobs
                                        )
        elif args.avgnum>1:
            entity_recognition_from_ground_truth_get_avg(args.avgnum,
                building_list=args.source_building_list,
                source_sample_num_list=args.sample_num_list,
                target_building=args.target_building,
                token_type='justseparate',
                label_type=args.label_type,
                use_cluster_flag=args.use_cluster_flag,
                use_brick_flag=args.use_brick_flag,
                eda_flag=args.eda_flag,
                ts_flag=args.ts_flag,
                negative_flag=args.negative_flag,
                n_jobs=args.n_jobs,
                worker_num=args.worker_num)
    elif args.prog == 'crf_entity':
        params = (args.source_building_list,
                  args.sample_num_list,
                  args.target_building,
                  'justseparate',
                  args.label_type,
                  args.use_cluster_flag,
                  args.use_brick_flag,
                  args.eda_flag,
                  args.negative_flag,
                  args.debug_flag,
                  args.n_jobs,
                  args.ts_flag)
        crf_entity_recognition_iteration(args.iter_num, args.postfix, *params)

        """
        entity_recognition_from_crf(\
                building_list=args.source_building_list,\
                source_sample_num_list=args.sample_num_list,\
                target_building=args.target_building,\
                token_type='justseparate',\
                label_type=args.label_type,\
                use_cluster_flag=args.use_cluster_flag,\
                use_brick_flag=args.use_brick_flag,\
                eda_flag=args.eda_flag,
                debug_flag=args.debug_flag,
                n_jobs=args.n_jobs)
        """
    elif args.prog == 'result':
        assert args.exp_type in ['crf', 'entity', 'crf_entity', 'entity_iter',
                                 'etc', 'entity_ts', 'cls']
        if args.exp_type == 'crf':
            crf_result()
        elif args.exp_type == 'entity':
            entity_result()
        elif args.exp_type == 'crf_entity':
            crf_entity_result()
        elif args.exp_type == 'entity_iter':
            entity_iter_result()
        elif args.exp_type == 'entity_ts':
            entity_ts_result()
        elif args.exp_type == 'cls':
            cls_comp_result()
        elif args.exp_type == 'etc':
            etc_result()

    elif args.prog == 'init':
        init()
    else:
        #print('Either learn or predict should be provided')
        assert(False)

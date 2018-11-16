import arrow
import pdb

from ir2tagsets_seq import Ir2Tagsets
from data_model import *
from exper import Exper
from eval_func import *

t0 = arrow.get()

connect('oracle')



column_names = ['VendorGivenName',
                 'BACnetName',
                 'BACnetDescription']

target_building = 'ebu3b'
source_buildings = ['ap_m', 'ebu3b']
source_sample_num_list = [200, 0]

building_sentence_dict = dict()
building_label_dict = dict()
building_tagsets_dict = dict()
for building in source_buildings:
    true_tagsets = {}
    label_dict = {}
    for labeled in LabeledMetadata.objects(building=building):
        srcid = labeled.srcid
        true_tagsets[srcid] = labeled.tagsets
        fullparsing = None
        for clm in column_names:
            one_fullparsing = [i[1] for i in labeled.fullparsing[clm]]
            if not fullparsing:
                fullparsing = one_fullparsing
            else:
                fullparsing += ['O'] + one_fullparsing
                #  This format is alinged with the sentence
                #  configormation rule.
        label_dict[srcid] = fullparsing

    building_tagsets_dict[building] = true_tagsets
    building_label_dict[building] = label_dict
    sentence_dict = dict()
    for raw_point in RawMetadata.objects(building=building):
        srcid = raw_point.srcid
        if srcid in true_tagsets:
            metadata = raw_point['metadata']
            sentence = None
            for clm in column_names:
                if not sentence:
                    sentence = [c for c in metadata[clm].lower()]
                else:
                    sentence += ['\n'] + \
                                [c for c in metadata[clm].lower()]
            sentence_dict[srcid]  = sentence
    building_sentence_dict[building] = sentence_dict

target_srcids = list(building_label_dict[target_building].keys())
t1 = arrow.get()
print(t1-t0)
inference_configs = {
    'n_jobs': 25,
    'negative_flag': True,
    'tagset_classifier_type': 'StructuredCC',
    'n_estimators': 20,
    'vectorizer_type': 'tfidf',
    'autoencode': False
}

ir2tagsets = Ir2Tagsets(target_building,
                        target_srcids,
                        building_label_dict,
                        building_sentence_dict,
                        building_tagsets_dict,
                        source_buildings,
                        source_sample_num_list,
                        conf=inference_configs
                        )
configs = {
    'step_num': 10,
    'iter_num': 25,
}
eval_functions = {
    'accuracy': get_accuracy_raw,
    'macro_f1': get_macro_f1_raw
}
exp = Exper(ir2tagsets, eval_functions, configs)
exp.run_exp()


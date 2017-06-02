import json

buildings = ['ebu3b', 'ap_m', 'bml']

postremover = lambda s: s.split('-')[0]
for building in buildings:
    with open('{0}_ground_truth.json'.format(building), 'r') as fp:
        data = json.load(fp)

    new_data = dict([(srcid, list(set((map(postremover, true_tagsets))))) for srcid, true_tagsets in data.items()])
    with open('{0}_true_tagsets.json'.format(building), 'w') as fp:
        json.dump(new_data, fp, indent=2)

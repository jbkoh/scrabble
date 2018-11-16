import json

with open('point_subclass_dict.json', 'r') as fp:
    d = json.load(fp)

found_tagsets = set()
redundant_tagsets = set()

for superclass, tagsets in d.items():
    redundant_tagsets.union(set([tagset for tagset in tagsets \
                                 if tagset in found_tagsets]))
    found_tagsets.union(set(tagsets))
print(redundant_tagsets)

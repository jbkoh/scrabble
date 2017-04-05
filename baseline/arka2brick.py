import random
from collections import OrderedDict
import pdb
from operator import itemgetter

import rdflib
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib import RDF, RDFS

BASE = Namespace("http://brickschema.org/example#")
BRICK = Namespace("http://brickschema.org/schema/brick#")
BF= Namespace("http://brickschema.org/schema/brickframe#")

def gen_index(ref):
    if ref:
        return ref
    else:
        return str(random.randint(0,100))

with open("Berkeley_SodaHall_MetadataParseKey", "r") as fp:
    building_dict = dict()
    for line in fp:
        line = line.replace(" ", "").replace("\t", "").replace("\n", "")
        srcid, line = line.split("=>")
        item_list = line.split(";")
        item_dict = dict()
        ref_dict = dict()
        for item in item_list:
            if item=="":
                continue
            key, val = item.split("->")
            if "Ref" in val:
                ref_dict[val] = key
            else:
                item_dict[key] = val
        building_dict[srcid] = (item_dict, ref_dict)

g = Graph()
g.bind('base', BASE)
g.bind('brick', BRICK)
g.bind('bf', BF)


weird_num = 0
zero_num = 0


# Topological Order
loc_order = ['zone', 'floor', 'site']
equip_order = [
        'supplyfan', 'exhaustfan', 'ahu', 
        'fancontrolunit',
        'condensorpump',
        'pump', 'coolingtower', 'airconditioner', 'chiller',
        'chilled/condensorwaterloop', 'hotwaterloop',
        ]
        
for srcid, (item_dict, ref_dict) in building_dict.items():
    point = BASE[srcid]
    # Add point type
    type_num = 0
    for item_tag, item_label in item_dict.items():
        if not item_label in ['zone', 'ahu', 'site', 'floor', 'supplyfan',
                'chilled/condensorwaterloop', 'chiller', 'pump', 'exhaustfan',
                'hotwaterloop', 'buildingwidesensor', 'airconditioner', 
                'fancontrolunit', 'coolingtower', 'condensorpump'
                ]:
            g.add((point, RDF.type, BRICK[item_label]))
            del item_dict[item_tag]
            type_num += 1
        if 'Sensor' in item_label or \
            'Setpoint' in item_label or \
            'Command' in item_label or \
            'Alarm' in item_label or \
            'Status' in item_label:
            g.add((point, RDF.type, BRICK[item_label]))
            type_num += 1
    if type_num > 1:
#        print(type_num, srcid)
        weird_num += 1
    if type_num == 0:
#        print(type_num, srcid)
        zero_num += 1

    # Add other info.    
    found_equip_dict = dict()
    found_loc_dict = dict()
    for item_tag, item_label in item_dict.items():
#        pdb.set_trace()
        other = BASE[item_label+gen_index(ref_dict.get(item_label+'Ref'))]
        g.add((other, RDF.type, BRICK[item_label]))
        if item_label in equip_order:
            found_equip_dict[equip_order.index(item_label)] = other 
        elif item_label in loc_order:
            found_loc_dict[loc_order.index(item_label)] = other
        else:
            print("no point no equip no loc, then what?: " + item_tag, \
                    item_label)
    found_equip_dict = OrderedDict(sorted(found_equip_dict.items(), 
                                    key=itemgetter(0)))
    found_equip_list = list(found_equip_dict.values())
    found_loc_dict = OrderedDict(sorted(found_loc_dict.items(), 
                                    key=itemgetter(0)))
    found_loc_list = list(found_loc_dict.values())

    # Add relationship mong other
    for i, equip in enumerate(found_equip_list):
        if i==0:
            continue
        prev_equip = found_equip_list[i-1]
        g.add((equip, BF.hasPart, prev_equip))
    for i, loc in enumerate(found_loc_list):
        if i==0:
            continue
        prev_loc = found_loc_list[i-1]
        g.add((loc, BF.hasLocation, prev_loc))

    g.add((point, BF.hasLocation, found_loc_list[0]))
    if len(found_equip_list)>0:
        g.add((found_equip_list[0], BF.hasPoint, point))

    #for item_tag, item_label in item_dict.items()
print weird_num, zero_num

g.serialize('temp.ttl', format='turtle')

# coding: utf-8

# In[45]:

from functools import reduce

import rdflib
from rdflib.namespace import RDFS
from rdflib import URIRef, BNode, Literal
import re
from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
import matplotlib.pyplot as plt

from termcolor import colored

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import OWL, RDF, RDFS
import rdflib
import json

from copy import deepcopy

uriPrefix = 'http://buildsys.org/ontologies/Brick#'

def lcs_len(X, Y):
	m = len(X)
	n = len(Y)
	# An (m+1) times (n+1) matrix
	C = [[0] * (n + 1) for _ in range(m + 1)]
	for i in range(1, m+1):
		for j in range(1, n+1):
			if X[i-1] == Y[j-1]: 
				C[i][j] = C[i-1][j-1] + 1
			else:
				C[i][j] = max(C[i][j-1], C[i-1][j])
	lenList = [subC[-1] for subC in C]
	return max(lenList)


#### Queries###
###############
subclassesQuery = lambda subclassName: ("""
        PREFIX rdf:<http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX brick: <http://buildsys.org/ontologies/Brick#>
        SELECT ?anything
        WHERE{
            ?anything rdfs:subClassOf+ %s.
        }
        """ %(subclassName)
		)



# In[46]:

BRICK      = Namespace('http://buildsys.org/ontologies/Brick#')
BRICKFRAME = Namespace('http://buildsys.org/ontologies/BrickFrame#')
BRICKTAG   = Namespace('http://buildsys.org/ontologies/BrickTag#')

BUILDING = Namespace('http://buildsys.org/ontologies/building_example#')

g = Graph()
print("Init Graph")
g.parse('../GroundTruth/Brick/Brick.ttl', format='turtle')
print("Load Brick.ttl")
g.parse('../GroundTruth/Brick/BrickFrame.ttl', format='turtle')
#g.parse('Brick/BrickTag.ttl', format='turtle')
print("Load BrickFrame.ttl")

g.bind('rdf'  , RDF)
g.bind('rdfs' , RDFS)
g.bind('brick', BRICK)
g.bind('bf'   , BRICKFRAME)
g.bind('btag' , BRICKTAG)


def printResults(res):
    if len(res) > 0:
        color = 'green'
    else:
        color = 'red'
    print(colored("-> {0} results".format(len(res)), color, attrs=['bold']))
    
def printTuples(res):
    for row in res:
        #print map(lambda x: x.split('#')[-1], row)
        print(row[0])
	
from collections import Counter
        
def extract_all_subclasses(g, subclassName, rawFlag=False):
	subclassList = list()
	res = g.query(subclassesQuery(subclassName))
	for row in res:
		thing = row[0]
		if rawFlag:
			subclass = thing.split('#')[-1]
		else:
			subclass = thing.split('#')[-1].lower()
		subclassList.append(subclass)
    
	try:
		assert(len(subclassList)==len(set(subclassList)))
	except:
		print('------------------------')
		print(len(subclassList))
		print(len(set(subclassList)))
		subclassCounter = Counter(subclassList)
		for subclass,cnt in subclassCounter.items():
			if cnt>1:
				print(subclass)
		print('------------------------')
		assert(False)
	return subclassList


equalDict = dict()
for s,p,o in g:
    if p==OWL.equivalentClass:
        a = s.split('#')[-1].lower()
        b = o.split('#')[-1].lower()
        equalDict[a] = b
        equalDict[b] = a



pointTagsetList = extract_all_subclasses(g, "brick:Alarm")+ \
				extract_all_subclasses(g, "brick:Command")+\
				extract_all_subclasses(g, "brick:Meter")+\
				extract_all_subclasses(g, "brick:Sensor")+\
				extract_all_subclasses(g, "brick:Status")+\
				extract_all_subclasses(g, "brick:Timer")+\
				extract_all_subclasses(g, "brick:Setpoint")
for pointTagset in pointTagsetList:
	if 'supply' in pointTagset:
		newPointTagset = pointTagset.replace('supply', 'discharge')
		if newPointTagset not in pointTagsetList:
			pointTagsetList.append(newPointTagset)
	if 'discharge' in pointTagset:
		newPointTagset = pointTagset.replace('discharge', 'supply')
		if newPointTagset not in pointTagsetList:
			pointTagsetList.append(newPointTagset)


equipTagsetList = extract_all_subclasses(g, "brick:Equipment")
locationTagsetList = extract_all_subclasses(g, "brick:Location")
measureTagsetList = extract_all_subclasses(g, "brick:MeasurementProperty")


# TODO: If it is worse, remove this
resourceTagsetList = extract_all_subclasses(g, "brick:Resource")

removingTagsetList = list()
usingAcronymList = ['hvac', 'vav', 'ahu', 'vfd', 'crac']
for tagset in pointTagsetList + equipTagsetList + locationTagsetList + measureTagsetList + resourceTagsetList:
	if tagset in equalDict.keys():
		if tagset in usingAcronymList:
			removingTagsetList.append(equalDict[tagset])
		else:
			#if len(tagset)<len(equalDict[tagset]) and lcs_len(tagset, equalDict[tagset])==len(tagset):
			if len(tagset)*2<len(equalDict[tagset]):
				removingTagsetList.append(tagset)
for tagset in removingTagsetList:
	try:
		pointTagsetList.remove(tagset)
	except:
		pass
	try:
		equipTagsetList.remove(tagset)
	except:
		pass
	try:
		locationTagsetList.remove(tagset)
	except:
		pass
	try:
		measureTagsetList.remove(tagset)
	except:
		pass
	try:
		ResourceTagsetList.remove(tagset)
	except:
		pass

for i, pointTagset in enumerate(pointTagsetList):
	if 'glycool' in pointTagset:
		del pointTagsetList[i]

tagsetList = pointTagsetList + equipTagsetList + locationTagsetList + measureTagsetList + resourceTagsetList
separater = lambda s:s.split('_')
tagList = list(set(reduce(lambda x,y: x+y, map(separater,tagsetList))))
equipTagList = list(set(reduce(lambda x,y: x+y, map(separater,equipTagsetList))))
pointTagList = list(set(reduce(lambda x,y: x+y, map(separater,pointTagsetList))))
locationTagList = list(set(reduce(lambda x,y: x+y, map(separater,locationTagsetList))))
measureTagList = list(set(reduce(lambda x,y: x+y, map(separater,measureTagsetList))))

if '' in tagList: tagList.remove('')


equipPointDict = dict()
origEquipTagsetList = extract_all_subclasses(g, "brick:Equipment", rawFlag=True)
for equipTagset in origEquipTagsetList:
	correspondingPointList = list()
	queryEquipTagset = ':'+equipTagset
	query = """
        PREFIX rdf:<http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX brick: <http://buildsys.org/ontologies/Brick#>
		PREFIX bf: <http://buildsys.org/ontologies/BrickFrame#>
        SELECT ?point
        WHERE{
			BIND (%s AS ?start)
			?start rdfs:subClassOf ?par .
			?par a owl:Restriction .
			?par owl:onProperty bf:hasPoint .
			?par owl:someValuesFrom ?point .
			}
	""" %queryEquipTagset
	res = g.query(query)
	for row in res:
		thing = row[0]
		point = thing.split('#')[-1].lower()
		point = point.split(':')[-1]
		correspondingPointList.append(point)
	if len(correspondingPointList)==0:
		continue
	equipTagsetName = equipTagset.lower()
	if not equipTagsetName in equipPointDict.keys():
		equipPointDict[equipTagsetName] = correspondingPointList
	if equipTagsetName in equalDict.keys():
		if not  equalDict[equipTagsetName] in equipPointDict.keys():
			equipPointDict[equalDict[equipTagsetName]] = correspondingPointList
for equipTagset in equipTagsetList:
	if not equipTagset in equipPointDict.keys():
		equipPointDict[equipTagset] = list()

for equipTagsetName, correspondingPointList in equipPointDict.items():
	for pointTagset in correspondingPointList:
		if 'supply' in pointTagset:
			newPointTagset = pointTagset.replace('supply', 'discharge')
			if newPointTagset not in correspondingPointList:
				correspondingPointList.append(newPointTagset)
		if 'discharge' in pointTagset:
			newPointTagset = pointTagset.replace('discharge', 'supply')
			if newPointTagset not in correspondingPointList:
				correspondingPointList.append(newPointTagset)



equalDict['co2_level_sensor'] = "co2_sensor"
equalDict['co2_sensor'] = "co2_level_sensor"
equalDict['room_temperature_setpoint'] = 'zone_temperature_setpoint'
equalDict['temperature_setpoint'] = 'zone_temperature_setpoint'
equalDict['zone_temperature_setpoint'] = 'temperature_setpoint'
equalDict['zone_temperature_setpoint'] = 'room_temperature_setpoint'
equalDict['temperature_setpoint'] = 'room_temperature_setpoint'
equalDict['room_temperature_setpoint'] = 'temperature_setpoint'
equalDict['effective_cooling_temperature_setpoint'] = 'cooling_temperature_setpoint'
equalDict['cooling_temperature_setpoint'] = 'effective_cooling_temperature_setpoint'
equalDict['effective_heating_temperature_setpoint'] = 'heating_temperature_setpoint'
equalDict['heating_temperature_setpoint'] = 'effective_heating_temperature_setpoint'


equipRelationDict = defaultdict(set)
equipRelationDict['ahu'] = set(['chilled_water_system','hot_water_system', 'heat_exchanger', 'economizer', 'supply_fan', 'return_fan', 'exhaust_fan', 'mixed_air_filter',\
						'mixed_air_damper', 'outside_air_damper', 'return_air_damper', 'cooling_coil', 'heating_coil', 'vfd'])
equipRelationDict['vav'] = set(['vav', 'reheat_valve', 'damper', 'booster_fan', 'vfd'])
equipRelationDict['supply_fan'] = set(['vfd', 'ahu'])
equipRelationDict['return_fan'] = set(['vfd', 'ahu'])
equipRelationDict['chilled_water_system'] = set(['chilled_water_pump', 'vfd', 'ahu'])
equipRelationDict['hot_water_system'] = set(['hot_water_pump', 'vfd', 'ahu'])

equipRelationDict['vfd'] = set(['supply_fan', 'return_fan', 'ahu', 'chilled_water_pump', 'hot_water_pump', 'chilled_water_system', 'hot_water_system', 'vav'])
equipRelationDict['chilled_water_pump'] = set(['chilled_water_system', 'ahu', 'vfd'])
equipRelationDict['hot_water_pump'] = set(['hot_water_system', 'ahu', 'vfd'])


for equip, subEquipList in list(equipRelationDict.items()):
	for subEquip in subEquipList:
		equipRelationDict[subEquip].add(equip)

equipRelationDict = dict(equipRelationDict)
for equip, subEquipList in equipRelationDict.items():
	equipRelationDict[equip] = list(subEquipList)

locationRelationDict = dict()
locationRelationDict['basement'] = ['room']
locationRelationDict['floor'] = ['room']
locationRelationDict['building'] = ['basement','floor','room']
locationRelationDict['room'] = ['basement', 'floor', 'building']


import csv
import numpy as np
import datetime
import time
from joblib import Parallel, delayed
import multiprocessing
import changefinder
import matplotlib.pyplot as plt
from feature_extractor import *
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import *
from sklearn import preprocessing
from random import shuffle
from sklearn.multiclass import *
from sklearn.externals import joblib
from sklearn.svm import *
from randomizer import select_random_samples
from sklearn import tree
from sklearn.preprocessing import normalize
from ploting_classification_report import plot_classification_report
import pickle
import pdb

class TimeSeriesToIR:

	def __init__(self, mlb=None, model=RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=2, random_state=0)):
		self.mlb = mlb
		self.model = model
		self.num_cores = multiprocessing.cpu_count()

	def get_binarizer(self):
		return self.mlb

	def train_model(self, features, cluster_filepath=None, training_percent=.4):

		temp = features

		index = range(len(temp))
		num_train = int(training_percent * float(len(index)))

		srcids = []

		for i in index:
			srcids.append(temp[i][2])

		if(cluster_filepath is None):
			X = list()
			Y = list()
			index = np.subtract(np.array(index), 1)
			for a in temp:
				X.append(normalize(np.array(a[0]).reshape((1,-1))).reshape(-1))
				Y.append(a[1][0])
			Y = np.array(Y)
			X = np.array(X)
			shuffle(index)
			X_train, X_test = X[index[:num_train]], X[index[num_train:]]
			Y_train, Y_test = Y[index[:num_train]], Y[index[num_train:]]
		else:
			X_train, X_test = list(), list()
			Y_train, Y_test = list(), list()
			randomness = select_random_samples(cluster_filename=cluster_filepath, srcids=srcids, n=num_train, use_cluster_flag=1)

			for i in range(len(index)):
				not_in_set = True
				for j in range(len(randomness)):
					if(temp[i][2] == randomness[j]):
						X_train.append(normalize(np.array(temp[i][0]).reshape((1,-1))).reshape(-1))
						Y_train.append(temp[i][1][0])
						not_in_set = False
				if(not_in_set):
					X_test.append(normalize(np.array(temp[i][0]).reshape((1,-1))).reshape(-1))
					Y_test.append(temp[i][1][0])
			X_train = np.array(X_train)
			X_test = np.array(X_test)
			Y_train = np.array(Y_train)
			Y_test = np.array(Y_test)
		print(X_train.shape)
		print(Y_train.shape)
		self.model.fit(X_train, Y_train)
		Y_pred = self.model.predict(X_test)
		report = classification_report(Y_test, Y_pred)
		plot_classification_report(report, self.mlb.classes_)

	def feature_extractor(self, srcids, schema_labels, data_path="data/", num_points=5000, save_path="temp_var.pkl"):
		features = Parallel(n_jobs=self.num_cores)(delayed(features)(self.mlb, srcids[x], schema_labels[x].lower().split(), data_path, num_points)
			for x in range(len(scrids)))

		with open(save_path, 'wb') as f:
			pickle.dump(features, f)

	def fit(self, train_features, train_srcids, train_tags_dict):
		X_train = []
		Y_train = []
		for srcid in train_srcids:
			for j in train_features:
				if(j[2] == srcid):
				#if(j[2].decode('utf-8') == srcid):
					X_train.append(normalize(np.array(j[0]).reshape((1,-1))).reshape(-1))
					#Y_train.append(j[1][0])
					Y_train.append(train_tags_dict[srcid])
		X_train = np.array(X_train)
		Y_train = self.mlb.transform(Y_train)
		self.model.fit(X_train, Y_train)

	def predict(self, test_features, test_srcids):
		test_data = []
		for i in test_srcids:
			for j in test_features:
				if(j[2] == i):
				#if(j[2].decode('utf-8') == i):
					test_data.append(normalize(np.array(j[0]).reshape((1,-1))).reshape(-1))

		Y_pred = self.model.predict(np.array(test_data))
		#temp_pred_proba = np.array(self.model.predict_proba(np.array(test_data)))
		#Y_proba = []
		#for i in range(temp_pred_proba.shape[1]):
	#		temp = []
#			for j in range(temp_pred_proba.shape[0]):
#				temp.append(1 - temp_pred_proba[j][i][0])
#			Y_proba.append(np.array(temp))
#		Y_proba = np.array(Y_proba)

		return Y_pred#, Y_proba

	def ts_to_ir (self, train_features, train_srcids, test_features, test_srcids):
		if(train_srcids is not None):
			X_train = []
			Y_train = []
			for i in train_srcids:
				for j in train_features:
					if(j[2] == i):
						X_train.append(normalize(np.array(j[0]).reshape((1,-1))).reshape(-1))
						Y_train.append(j[1][0])
			X_train = np.array(X_train)
			Y_train = np.array(Y_train)
			print(X_train.shape)
			print(Y_train.shape)
			self.model.fit(X_train, Y_train)

		if(test_srcids is not None):
			test_data = []
			for i in test_srcids:
				for j in test_features:
					if(j[2] == i):
						test_data.append(normalize(np.array(j[0]).reshape((1,-1))).reshape(-1))

			Y_pred = self.model.predict(np.array(test_data))
			temp_pred_proba = np.array(self.model.predict_proba(np.array(test_data)))
			Y_proba = []
			for i in range(temp_pred_proba.shape[1]):
				temp = []
				for j in range(temp_pred_proba.shape[0]):
					temp.append(1 - temp_pred_proba[j][i][0])
				Y_proba.append(np.array(temp))
			Y_proba = np.array(Y_proba)

			return self.mlb.classes_, Y_pred, Y_proba


def features(mlb, srcid, data_labels, file_path="data/", num_points=None):
	reader = csv.reader(open(file_path + srcid +".csv", "rb"), delimiter=",")
	file_data = list(reader)
	temp = list()
	for y in file_data[1:]:
		temp.append(y[1])
	data = np.array(temp, dtype="float")
	if(data_labels is None):
		Y = None
	else:
		Y = mlb.transform([set(data_labels)])

	if num_points is None:
		features = get_features(data)
	elif num_points >= len(data):
		features = get_features(data)
	else:
		features = get_features(data[-num_points:])

	return features, Y, srcid


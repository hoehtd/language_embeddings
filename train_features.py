import sys
import os
import csv
import collections
import pickle

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy import stats
import sklearn.decomposition

os.environ['CUDA_VISIBLE_DEVICES'] = ''

def e_insensitive_loss(y_true, y_pred):
	return K.relu(K.abs(y_true - y_pred) - .1);

def linreg_model(n_cats = 1, activation = 'sigmoid'):
	ipt = keras.layers.Input(shape = (50,))
	l1 = keras.layers.Dense(2, activation = 'tanh')(ipt)
	y = keras.layers.Dense(n_cats, activation = activation)(l1)
	# y = keras.layers.Dense(n_cats, activation = 'sigmoid')(ipt)
	model = keras.Model(inputs = ipt, outputs = y)
	return model

def load_embeddings(file, normalize = True):
	word_embs = dict()
	print(f'Loading word embeddings from {file}')
	with open(file) as ein:
		# ein.readline()
		for line in ein:
			lp = line.rstrip('\n ').split(' ')
			word_embs[lp[0]] = np.asarray([float(x) for x in lp[1:]])
			if normalize:
				word_embs[lp[0]] = word_embs[lp[0]] / np.linalg.norm(word_embs[lp[0]])

	return word_embs


def train_field(embs, y, save_name):
	
	embsmat = list([embs[l] for l in y])
	yarr = list([y[l] for l in y])

	model = linreg_model()
	optm = keras.optimizers.Adam(lr = 5e-2)
	model.compile(optimizer = optm, loss = e_insensitive_loss, metrics = ['mae'])
	data = np.asarray(embsmat)
	model.fit(data, yarr, verbose = 1, epochs = 50)

	model.save(save_name+'.h5')

	return model

def train_field_categorical(embs, y, save_name):
	
	embsmat = list([embs[l] for l in y])
	ldata = len(embsmat)
	yarr = K.one_hot([y[l] for l in y], 2)

	model = linreg_model(2, 'softmax')
	optm = keras.optimizers.Adam(lr = 5e-2)
	model.compile(optimizer = optm, loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
	data = np.asarray(embsmat)
	model.fit(data, yarr, verbose = 1, epochs = 50, steps_per_epoch = ldata)

	model.save(save_name+'.h5')

	return model

def train_all(langembs, data_loc):
	with open(data_loc, 'rb') as pin:
		[langs, features, data_mat] = pickle.load(pin)
	selected_langs = dict()
	for i, l in enumerate(langs):
		if l in langembs:
			selected_langs[l] = i
	# print_data(langs, features, data_mat, 'dependencies_direction.csv')
	results = []
	predictions = []
	for i in range(len(features)):
		keras.backend.clear_session()

		print(f'Dimension {i}')
		lresult = []
		lpreds = []
		for hol in selected_langs:
			hol_i = selected_langs[hol]
			print('Predicting {}'.format(hol))
			y = dict()
			for l in selected_langs.values():
				if l != hol_i:
					y[langs[l]] = data_mat[l][i]
			
			model = train_field(langembs, y, 'models/'+features[i]+'_'+hol)
			res = model.evaluate(np.expand_dims(langembs[hol], 0), [data_mat[hol_i][i]])
			pred = model.predict(np.expand_dims(langembs[hol], 0))[0][0]
			lpreds.append(pred)
			lresult.append(res[0])

		with open(f'models/{features[i]}_results.txt', 'w') as fout:
			fout.write(' '.join(langembs.keys())+'\n')
			fout.write(' '.join([str(x) for x in lresult]) + '\n')

		results.append(lresult)
		predictions.append(lpreds)

	print_data(selected_langs.keys(), features, np.transpose(predictions), 'models/model_predictions.csv')
	with open('models/results.csv', 'w') as fout:
		fout.write('lang,'+','.join(['"'+s+'"' for s in features])+'\n')
		results = np.transpose(results)
		for i,l in enumerate(list(selected_langs.keys())):
			fout.write(l)
			fout.write(','+','.join([str(x) for x in results[i]])+'\n')

def train_all_categorical(langembs, data_loc):
	with open(data_loc, 'rb') as pin:
		[langs, features, data_mat] = pickle.load(pin)
	
	for i in range(len(data_mat)):
		for j in range(len(data_mat[i])):
			if data_mat[i][j] > .5:
				data_mat[i][j] = 1
			else:
				data_mat[i][j] = 0

	# print_data(langs, features, data_mat, 'dependencies_direction_bin.csv')
	results = []
	predictions = []
	for i in range(len(features)):
		keras.backend.clear_session()

		print(f'Dimension {i}')
		lresult = []
		lpreds = []
		for hol_i, hol in enumerate(langs):
			print('Predicting {}'.format(hol))
			y = dict()
			for l in range(len(langs)):
				if l != hol_i:
					y[langs[l]] = data_mat[l][i]
			
			model = train_field_categorical(langembs, y, 'models/'+features[i]+'_'+hol)
			res = model.evaluate(np.expand_dims(langembs[hol], 0), K.eval(K.one_hot([data_mat[hol_i][i]], 2)))
			pred = model.predict(np.expand_dims(langembs[hol], 0))
			# print(pred, res)
			lpreds.append(np.argmax(pred[0]))
			lresult.append(res[1])

		with open(f'models/{features[i]}_results.txt', 'w') as fout:
			fout.write(' '.join(langembs.keys())+'\n')
			fout.write(' '.join([str(x) for x in lresult]) + '\n')

		results.append(lresult)
		predictions.append(lpreds)

	print_data(langs, features, np.transpose(predictions), 'models/model_predictions.csv')
	with open('models/results.csv', 'w') as fout:
		fout.write('lang,'+','.join(['"'+s+'"' for s in features])+'\n')
		results = np.transpose(results)
		for i, l in enumerate(langembs):
			fout.write(l)
			fout.write(','+','.join([str(x) for x in results[i]])+'\n')

def print_data(x_axis, y_axis, mat, fname):
	with open(fname, 'w') as fout:
		fout.write(',' + ','.join(['"'+y+'"' for y in y_axis]) + '\n')
		for i, x in enumerate(x_axis):
			fout.write(x+','+','.join([str(n) for n in mat[i]]) + '\n')

if __name__ == '__main__':
	langembs = load_embeddings('../wals/seed_lm_2_unk_380k_feat.vec')

	train_all(langembs, 'all_common_relations.pickle')

import sys
import os
import csv
import collections
import random

import numpy as np
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster import hierarchy
from sklearn import metrics, cluster, decomposition
from adjustText import adjust_text

from linreg import load_embeddings, load_wals


def sidebyside_pca(names, embs, cats1, cats2):
	colors = ['#0F60FF','#02CEE8','#E81809','#14E802','#D1FF0F','#FFDC03','#E89909','#FF6103','#0FFF8E','#FF03D1']
	pca = decomposition.PCA(n_components = 2)
	red = pca.fit_transform(embs)
	fig,ax = plt.subplots(1, 2, figsize = (30,15))
	X = red[:,0]
	Y = red[:,1]
	ax[0].scatter(X, Y, c = [colors[l] for l in cats1])
	ax[1].scatter(X, Y, c = [colors[l] for l in cats2])

	for i, n in enumerate(names):
		ax[0].annotate(n, (X[i],Y[i]))
		ax[1].annotate(n, (X[i],Y[i]))
	fig.tight_layout()

	# patches = [mpatches.Patch(color = cmap[g], label = g) for g in cmap]
	# plt.legend(handles = patches)
	plt.show()


def colorshape_pca(names, embs, cats1, cats2, legend1):
	colors = ['#0F60FF','#02CEE8','#E81809','#14E802','#FF03D1','#FFDC03','#E89909','#FF6103','#0FFF8E','#D1FF0F']
	shapes = ['o', 's', '^', '*', 'P', 'D', '2', 'h', 'x', '+']
	pca = decomposition.PCA(n_components = 2)
	red = pca.fit_transform(embs)
	fig,ax = plt.subplots(figsize = (12,6))
	X = red[:,0]
	Y = red[:,1]
	for i in range(len(X)):
		ax.scatter(X[i], Y[i], s = 75, c = colors[cats1[i]], marker = shapes[cats2[i]])

	annotations = []
	for i, n in enumerate(names):
		annotations.append(ax.text(X[i], Y[i], n, fontsize = 15))
	# adjust_text(annotations)
	# adjust_text(annotations)
	# adjust_text(annotations)

	fig.tight_layout()

	patches = [mpatches.Patch(color = colors[i], label = legend1[i]) for i in range(len(legend1))]
	plt.legend(handles = patches)
	plt.show()
	# plt.savefig('colorshape_pca.svg', pad_inches = 0)


def k_means(embs, categories):
	ncats = len(set(categories.values()))
	X = []
	Y = []
	for w in embs:
		X.append(embs[w])
		Y.append(categories[w])
	kmeans = cluster.KMeans(n_clusters = 4, n_jobs = -1, n_init = 50, max_iter = 1000).fit(X)
	print(metrics.adjusted_rand_score(Y, kmeans.labels_))
	return kmeans.labels_

def spectral_cluster(embs, categories):
	ncats = len(set(categories.values()))
	X = []
	Y = []
	for w in embs:
		X.append(embs[w])
		Y.append(categories[w])
	spectral = cluster.SpectralClustering(n_clusters = 4, n_jobs = -1, n_init = 50).fit(X)
	print(metrics.adjusted_rand_score(Y, spectral.labels_))
	return spectral.labels_

def centroid_distances(embs, categories):
	pass


def random_pred(langembs, walsmat, n_cats):
	random.seed()
	results = []
	for i in range(n_cats):
		lresult = []
		for hol in langembs:
			y = []
			for l in langembs:
				if l != hol and walsmat[l][i]!='_':
					y.append(walsmat[l][i])

			if walsmat[hol][i] in y:
				lbl = random.choice(y)
				if lbl == walsmat[hol][i]:
					lresult.append(1)
				else:
					lresult.append(0)
			else:
				lresult.append(-3)

		results.append(lresult)
	results = np.transpose(results)
	return results

def plot_dendrogram(langembs):
	langs = list(langembs.keys())
	embs = [langembs[l] for l in langs]
	langs = [wals2eng[l] for l in langs]
	Z = hierarchy.linkage(embs, 'ward')
	fig,ax = plt.subplots(figsize = (13,6))
	# fig.tight_layout()
	dng = hierarchy.dendrogram(Z, ax = ax, labels = langs)
	plt.show()

def area_accuracies(results, permtest = 0):
	amap = dict()
	areas = set()
	with open('wals_category_areas.txt') as fin:
		for line in fin:
			wc, a = line.strip().split('\t')
			amap[wc] = a
			areas.add(a)
	areas = sorted(list(areas))

	with open(results) as cin:
		creader = csv.reader(cin)
		header = []
		mat = []
		for row in creader:
			if len(header) == 0:
				header = row[1:]
			else:
				mat.append([float(x) for x in row[1:]])

	def _get_col_acc(col):
		total = 0
		correct = 0
		for i in col:
			if i >= 0:
				total += 1
				correct += i
		return correct/total

	def _get_area_accs(mat, header):
		area_acc = dict()
		for i in areas:
			area_acc[i] = [0,0]
		mat = np.asarray(mat)
		for i,wcat in enumerate(header):
			curr = wcat.split()[0]
			ccat = amap[curr]
			acc = _get_col_acc(mat[:,i])
			area_acc[ccat][0]+=acc
			area_acc[ccat][1]+=1
		return area_acc

	if permtest > 0:
		wiki2wals = {'en': 'eng', 'ar': 'ams', 'bg': 'bul', 'ca': 'ctl', 'hr': 'scr', 'cs': 'cze', 'da': 'dsh', 'nl': 'dut', 'et': 'est', 'fi': 'fin', 'fr': 'fre', 'de': 'ger', 'el': 'grk', 'he': 'heb', 'hu': 'hun', 'id': 'ind', 'it': 'ita', 'no': 'nor', 'pl': 'pol', 'pt': 'por', 'ro': 'rom', 'ru': 'rus', 'sk': 'svk', 'sl': 'slo', 'es': 'spa', 'sv': 'swe', 'tr': 'tur', 'uk': 'ukr', 'vi': 'vie'}
		langs = list(wiki2wals.values())
		walsheader, walsmat = load_wals('languages_all_fields.csv', langs, .5)

		precord = dict()
		for i in areas:
			precord[i] = []
		for i in range(permtest):
			sys.stdout.write(f'\r{i+1}/{permtest}')
			pmat = random_pred(langs, walsmat, len(walsheader))
			pacc = _get_area_accs(pmat, walsheader)
			for ar in areas:
				if pacc[ar][1]!=0:
					precord[ar].append(pacc[ar][0]/pacc[ar][1])
			if i == 0:
				nacc = dict()
				for ar in precord:
					if precord[ar] != []:
						nacc[ar] = precord[ar]
				precord = nacc

		print()
		ob_acc = _get_area_accs(mat, header)
		res = dict()
		for ar in precord:
			obs = ob_acc[ar][0] / ob_acc[ar][1]
			res[ar] = (ar, obs, stats.percentileofscore(precord[ar], obs, kind = 'strict'))
			print('{}:\tmean: {}\tpercentile:{}'.format(*res[ar]))
	else:
		area_acc = _get_area_accs(mat, header)
		l1 = []
		l2 = []
		for ar in areas:
			if area_acc[ar][1] != 0:
				# print(f'{ar}:\t{area_acc[ar][0]/area_acc[ar][1]}')
				l1.append(ar)
				l2.append(str(area_acc[ar][0]/area_acc[ar][1]))
		print('\t'.join(l1))
		print('\t'.join(l2))

def main():
	global wals2eng
	wiki2wals = {'en': 'eng', 'ar': 'ams', 'bg': 'bul', 'ca': 'ctl', 'hr': 'scr', 'cs': 'cze', 'da': 'dsh', 'nl': 'dut', 'et': 'est', 'fi': 'fin', 'fr': 'fre', 'de': 'ger', 'el': 'grk', 'he': 'heb', 'hu': 'hun', 'id': 'ind', 'it': 'ita', 'no': 'nor', 'pl': 'pol', 'pt': 'por', 'ro': 'rom', 'ru': 'rus', 'sk': 'svk', 'sl': 'slo', 'es': 'spa', 'sv': 'swe', 'tr': 'tur', 'uk': 'ukr', 'vi': 'vie'}
	wals2eng = {'eng' : 'English', 'ams' : 'Arabic', 'bul' : 'Bulgarian', 'ctl' : 'Catalan', 'scr' : 'Croatian', 'cze' : 'Czech', 'dsh' : 'Danish', 'dut' : 'Dutch', 'est' : 'Estonian', 'fin' : 'Finnish', 'fre' : 'French', 'ger' : 'German', 'grk' : 'Greek', 'heb' : 'Hebrew', 'hun' : 'Hungarian', 'ind' : 'Indonesian', 'ita' : 'Italian', 'nor' : 'Norwegian', 'pol' : 'Polish', 'por' : 'Portuguese', 'rom' : 'Romanian', 'rus' : 'Russian', 'svk' : 'Slovak', 'slo' : 'Slovene', 'spa' : 'Spanish', 'swe' : 'Swedish', 'tur' : 'Turkish', 'ukr' : 'Ukrainian', 'vie' : 'Vietnamese'}
	langembs = load_embeddings('../langembs/29lang_matched_langembs_dec.vec')
	nl = dict()
	for l in wiki2wals:
		# if l not in {'el', 'hu', 'vi', 'id', 'tr'}:
		if l in langembs:
			nl[wiki2wals[l]] = langembs[l]
	langembs = nl

	names = list(langembs.keys())
	header, walsd = load_wals('languages_all_fields.csv', names, startfrom = 6)
	genus = dict()
	g2id = dict()
	for w in names:
		if walsd[w][0] not in g2id:
			g2id[walsd[w][0]] = len(g2id)
		genus[w] = g2id[walsd[w][0]]

	pred = spectral_cluster(langembs, genus)
	# sidebyside_pca(names, [langembs[w] for w in names], pred, [genus[w] for w in names])
	colorshape_pca([wals2eng[n] for n in names], [langembs[w] for w in names], [genus[w] for w in names], pred, [y[0] for y in sorted(g2id.items(), key = lambda x: x[1])])
	# plot_dendrogram(langembs)


if __name__ == '__main__':
	main()
	# area_accuracies('seed_lm_2_models/results.csv', permtest = 0)
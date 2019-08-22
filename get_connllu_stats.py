import sys
import os
import csv
import collections
import pickle


def proc_file(file, relations):
	if relations is None:
		relations = dict()
	
	with open(file) as fin:
		for line in fin:
			if line.startswith('#') or line.startswith('\n'):
				continue
			fields = line.split()
			try:
				curr = float(fields[0])
				root = float(fields[6])
			except ValueError:
				continue
			rel = fields[7]
			if rel == '_' or root == 0:
				continue
			if rel not in relations:
				relations[rel] = [0,0]
			relations[rel][1] += 1
			if root > curr:
				relations[rel][0] += 1

	return relations

def get_all_relations(conlluloc, langs):
	all_rels = dict()
	for r, d, f in os.walk(conlluloc):
		fds = d
		break

	for l in langs:
		files = []
		for dr in fds:
			if dr.startswith(f'UD_{langs[l]}'):
				for r,d,fs in os.walk(os.path.join(conlluloc, dr)):
					for fn in fs:
						if fn.endswith('train.conllu'):
							files.append(os.path.join(conlluloc, dr, fn))
	
		lrels = dict()
		for f in files:
			lrels = proc_file(f, lrels)
		all_rels[l] = lrels
	
	commonrels = set.intersection(*[set(all_rels[l].keys()) for l in all_rels])
	commonrels = list(commonrels)
	frozen_langs = []
	mat = []

	for l in langs:
		frozen_langs.append(l)
		rels = []
		for r in commonrels:
			rels.append(all_rels[l][r][0] / all_rels[l][r][1])
		mat.append(rels)

	with open('all_common_relations.pickle', 'wb') as pout:
		pickle.dump([frozen_langs, commonrels, mat], file = pout)
	# print(frozen_langs, commonrels, mat)

if __name__ == '__main__':
	wiki2lang = {'en': 'English', 'ar': 'Arabic', 'bg': 'Bulgarian', 'ca': 'Catalan', 'hr': 'Croatian', 'cs': 'Czech', 'da': 'Danish', 'nl': 'Dutch', 'et': 'Estonian', 'fi': 'Finnish', 'fr': 'French', 'de': 'German', 'el': 'Greek', 'he': 'Hebrew', 'hu': 'Hungarian', 'id': 'Indonesian', 'it': 'Italian', 'no': 'Norwegian-Bokmaal', 'pl': 'Polish', 'pt': 'Portuguese', 'ro': 'Romanian', 'ru': 'Russian', 'sk': 'Slovak', 'sl': 'Slovenian', 'es': 'Spanish', 'sv': 'Swedish', 'tr': 'Turkish', 'uk': 'Ukrainian', 'vi': 'Vietnamese'}
	get_all_relations('../conll_2018/ud-treebanks-v2.2/', wiki2lang)
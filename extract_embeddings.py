import sys
import re

import torch
import numpy

def extract_embeddings(model, emb_type = 'feat', layer = 'dec'):
	if emb_type == 'word':
		seq_num = 0
	elif emb_type == 'feat':
		seq_num = 1
	else:
		print('Parameter error: emb_type')
		return
	if layer not in {'enc', 'dec'}:
		print('Parameter error: layer')
		return
	model_name = model.split('/')[-1].split('.')[0]
	model = torch.load(model, map_location = 'cpu')
	layers = model['model'].keys()
	embeddings = None
	for lname in layers:
		if re.fullmatch(f'{layer}oder\.embeddings.*{seq_num}.*', lname):
			# print(lname)
			embeddings = model['model'][lname].numpy()
	if embeddings is None:
		print('embeddings not found')
		return

	if layer == 'dec':
		vdir = 'tgt'
	else:
		vdir = 'src'
	vocab = model['vocab'][vdir][seq_num][1].vocab.stoi
	with open(f'{model_name}_extracted_{layer}_{emb_type}.vec', 'w') as fout:
		for tok in vocab:
			fout.write(tok+' ')
			fout.write(' '.join([str(x) for x in embeddings[vocab[tok]]]))
			fout.write('\n')


if __name__ == '__main__':
	extract_embeddings(*sys.argv[1:])
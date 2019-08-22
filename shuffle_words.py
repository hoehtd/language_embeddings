import sys
import random


random.seed()
with open(sys.argv[1]) as fin:
	with open(sys.argv[2], 'w') as fout:
		for line in fin:
			wds = line.rstrip().split()
			random.shuffle(wds)
			fout.write(' '.join(wds)+'\n')
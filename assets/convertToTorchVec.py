import torch
import numpy as np 
import gensim
from gensim.models import Word2Vec 
from gensim.test.utils import common_texts, get_tmpfile
import collections

import json

model = gensim.models.KeyedVectors.load_word2vec_format('assets/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin', binary=True, limit=500000)

def convertToTorchVec(bdict):
	w2v = []
	for key in bdict:
		for word in bdict[key]:
			w2v.append(model.wv[word])
	print (torch.Tensor(w2v).shape)
	return torch.Tensor(w2v)

def convertToTorchDict(bdict):
	w2v = collections.defaultdict(list)
	for key in bdict:
		for word in bdict[key]:
			curr_word2vec = model.wv[word]
			w2v[key].append(curr_word2vec)
	tsr = {}
	for key in w2v:
		tsr[key] = torch.Tensor(w2v[key])
	
	for key in tsr:
		print (key)
		print (tsr[key].shape)
	return tsr	




def readBoards(fileName = 'assets/board_list.json'):
	with open(fileName, 'r') as f:
		dicts = []
		for line in f:
			board = json.loads(line)
			tensorDict = convertToTorchVec(board) 
			dicts.append(tensorDict)
	

# print (dicts)
readBoards()
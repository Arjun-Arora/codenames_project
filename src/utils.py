import scipy as sp 
import torch
import numpy as np 
import gensim
from gensim.models import Word2Vec 
from gensim.test.utils import common_texts, get_tmpfile
import collections
import argparse
import json

# parser = argparse.ArgumentParser(description='generate graph based on kNN')
# parser.add_argument('--boardfile', '-o', help='file of boards', default='assets/board_list.json')
# args = parser.parse_args()

model = gensim.models.KeyedVectors.load_word2vec_format('./assets/GoogleNews-vectors-negative300.bin', binary=True, limit=500000)

def findNearestWord(corpus,model,vector): 
	'''
	find nearest word function: 
	inputs:
		corpus: List of words that the corpus we're using for this codenames game
		model: this is the gensim word2vec model we're using for this function
		vector: numpy array representing a chosen word2vecVector 

	Output: String word that is nearest to word2vecVector from corpus (cosine distance)
	'''

	vecCorpus = sp.asarray([model.wv[word] for word in corpus])
	minIndex = sp.argmin([sp.spatial.distance.cosine(vecCorpus[i],vector) for i in range(len(vecCorpus))],axis = 0) #minimize across words not across vector length
	return corpus[minIndex]


def convertToTorchDict(bdict):
	w2v = collections.defaultdict(list)
	for key in bdict:
		for word in bdict[key]:
			curr_word2vec = model.wv[word]
			w2v[key].append(curr_word2vec)
	tsr = {}
	for key in w2v:
		tsr[key] = torch.Tensor(w2v[key])
	
	# for key in tsr:
	# 	print (key)
	# 	print (tsr[key].shape)
	return tsr

def readWordBoards(fileName="./assets/gensim_dev_board_list.json"):
	with open(fileName,'r') as f:
		dicts = []
		for line in f:
			board = json.loads(line)
			dicts.append(board)
	return dicts
#


def readBoards(fileName = "./assets/gensim_board_list.json"):
	with open(fileName, 'r') as f:
		dicts = []
		for line in f:
			board = json.loads(line)
			tensorDict = convertToTorchDict(board) 
			dicts.append(tensorDict)
	return dicts
	


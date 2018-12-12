#pass word dictionary
#pass glovevec from word
#loss function(params) -> scalar
#takes in embedding, outputs word (nearest neighbor)

import json
import numpy as np
import scipy as sp

def load_glove_model(glove_file="assets/glove.6B.50d.txt"):
    # print "Loading Glove Model"
    f = open(glove_file,'r')
    model = {}
    w =  open("assets/word_list.txt", 'r')
    for line in f:
        splitLine = line.split()
        word = splitLine[0]

        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    # print "Done.",len(model)," words loaded!"
    return model


def get_boards():
    file = "assets/board_list.json"
    with open(file, 'r') as f:
        dicts = []
        for line in f:
            board = json.loads(line)
            dicts.append(board)
    return dicts

def find_nearest_word(corpus, vector, words_to_avoid): 
	'''
	find nearest word function: 
	inputs:
		corpus: List of words that the corpus we're using for this codenames game
		model: this is the gensim word2vec model we're using for this function
		vector: numpy array representing a chosen word2vecVector 

	Output: String word that is nearest to word2vecVector from corpus (cosine distance)
	'''
	words_to_avoid = [str(word) for word in words_to_avoid]
	# print corpus.keys()[:10]
	min_word = ""
	min_dist = float('Inf')
	for word in corpus:
		if word not in words_to_avoid:
			curr_dist = sp.spatial.distance.cosine(corpus[word], vector)
			if curr_dist < min_dist:
				min_word = word
				min_dist = curr_dist

	# min_idx = sp.argmin([sp.spatial.distance.cosine(corpus[word],vector) for word in corpus if word not in words_to_avoid]) #minimize across words not across vector length
	# min_word = corpus.keys()[min_idx]
	return min_word

# if __name__ == "__main__":
	# with open("assets/word_list.txt", 'r') as fin:
	# 	with open("assets/glove_list.txt", "w+") as fout:
	# 		old_words = fin.read().split()
	# 		for w in old_words:
	# 			neww = w.replace("_", "")
	# 			neww = neww.lower()
	# 			fout.write(neww + "\n")
	# model = load_glove_model()
	# with open("assets/glove_list.txt", 'r') as f:
	# 	words = f.read().split()
	# 	for word in words:
	# 		if word not in model:
	# 			print (word)

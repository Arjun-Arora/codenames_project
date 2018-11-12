import scipy as sp 
from gensim.models import Word2Vec

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



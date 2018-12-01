import scipy as sp
import torch
import torch.nn.functional as F
import copy
import numpy as np
import itertools

def createSequential(board_sz, w2v_sz, k, n=4,h=300):
	modules = []
	activation = torch.nn.modules.activation.ReLU()
	l = torch.nn.Linear(board_sz*w2v_sz,h)
	for i in range(n):
		l = torch.nn.Linear(h,h)
		#linear layer, then nonlinear activation layer 
		modules.append(l)
		modules.append(activation)

	l = torch.nn.Linear(h,k)
	modules.append(l)
	activation = torch.nn.Softmax() #generalization of sigmoid function - sums, then normalizes so probs sum to 1 
	modules.append(activation)

	sequential = torch.nn.Sequential(*modules)

	return sequential

def BasicLoss(probDistribution,boardDict,w2vDict,assassinWeight=2.0): #boardDict is word2vec Tensor version of board with labels 
	probDistribution = torch.sum(probDistribution, dim=0) 
	# nx = probDistribution.detach().numpy()
	# max_prob = max(nx)
	# idx = np.where(nx == max_prob)[0][0]
	# print (probDistribution)
	completeLoss = 0
	for idx, prob in enumerate(probDistribution):
		outputVector = w2vDict[idx]
		sumLoss = 0 
		for key in boardDict:
			currLoss = 0
			if(key == 'blue'):
				blueTensor = boardDict[key]
				for word2vec in blueTensor:
					#print("word2vecShape: {},outputVectorShape:{}".format(word2vec.shape,outputVector.shape))
					#for every individual word2vec vector, sum cosine similarity
					currLoss += 1 - F.cosine_similarity(word2vec,outputVector,dim=0) #cosine dist = 1-cosine_similarity
				currLoss = -torch.mean(currLoss)
			elif(key == 'red'):
				redTensor = boardDict[key]
				for word2vec in redTensor:
				 	#for every individual word2vec vector, sum cosine similarity
				 	currLoss += 1 - F.cosine_similarity(word2vec,outputVector,dim=0) #cosine dist = 1-cosine_similarity
				currLoss = torch.mean(currLoss)
			elif(key =='assassin'):
				assassinVector = boardDict[key] #should be 1,len(word2Vec)
				currLoss = (torch.mean(assassinWeight*(1-F.cosine_similarity(assassinVector,outputVector,dim=0))))
			# print(currLoss.shape)
			sumLoss += currLoss
		
		completeLoss += prob * sumLoss
	# print (sumLoss)
	# print (completeLoss)
	# print ("\n\n\n\n\n")
	return completeLoss

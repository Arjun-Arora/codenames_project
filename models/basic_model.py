import scipy as sp
import torch
import torch.nn.functional as F
import copy
import itertools

class BasicModel(torch.nn.Module):
	def __init__(self,board_sz, w2v_sz, n=4,h=200):
		super(BasicModel,self).__init__()
		self.module_list = []
		activation = torch.nn.modules.activation.ReLU()
		l = torch.nn.Linear(board_sz * w2v_sz, h)
		self.module_list.append(l)
		for i in range(n):
			l = torch.nn.Linear(h, h)
			# linear layer, then nonlinear activation layer
			self.module_list.append(l)
			self.module_list.append(activation)
		l = torch.nn.Linear(h, w2v_sz)
		self.module_list.append(l)
		self.module_list.append(activation)
		self.layers = torch.nn.Sequential(*self.module_list)
	def forward(self,x):
		return self.layers(x)

def BasicLoss(outputVector,boardDict,assassinWeight=2.0,redWeight=0.1): #boardDict is word2vec Tensor version of board with labels
	sumLoss = 0 
	for key in boardDict:
		currLoss = 0
		if(key == 'blue'):
			blueTensor = boardDict[key]
			for word2vec in blueTensor:
				#print("word2vecShape: {},outputVectorShape:{}".format(word2vec.shape,outputVector.shape))
				#for every individual word2vec vector, sum cosine similarity
				currLoss += F.cosine_similarity(word2vec,outputVector,dim=0) #cosine dist = 1-cosine_similarity
			currLoss = -(torch.mean(currLoss))
			# print(currLoss)
		# elif(key == 'red'): #move away from red word
		# 	redTensor = boardDict[key]
		# 	for word2vec in redTensor:
		# 	 	#for every individual word2vec vector, sum cosine similarity
		# 	 	currLoss += F.cosine_similarity(word2vec,outputVector,dim=0) #cosine dist = 1-cosine_similarity
		# 	currLoss = redWeight*(torch.mean(currLoss))
		# elif(key =='assassin'): #want to move away from assassin word
		# 	assassinVector = boardDict[key] #should be 1,len(word2Vec)
		# 	currLoss = (torch.mean(assassinWeight*(F.cosine_similarity(assassinVector,outputVector,dim=0))))
		# print(currLoss.shape)
		sumLoss += currLoss
	return sumLoss

def KWordLoss(outputVector,boardDict,assassinWeight=2.0, k=1):
	minLoss = float('Inf')
	kBoardDict = copy.deepcopy(boardDict)
	for combo in itertools.combinations(boardDict['blue'], k):
		
		kBoardDict['blue'] = combo
		# print (combo)
		loss = BasicLoss(outputVector,kBoardDict,assassinWeight)
		minLoss = min(loss, minLoss)
	return minLoss
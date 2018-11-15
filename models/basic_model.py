import scipy as sp
import torch
import torch.nn.functional as F

class BasicModel(torch.nn.Module):
	def __init__(self,Data_in,H,Output):
		super(BasicModel,self).__init__()
		self.linear1 = torch.nn.Linear(Data_in,H)
		self.linear2 = torch.nn.Linear(H,Output)

	def forward(self,x):
		h_relu = self.linear1(x).clamp(min=0)
		outputVector = self.linear2(h_relu)
		outputVector = outputVector.mean(dim=1)
		return outputVector
def BasicLoss(outputVector,boardDict,assassinWeight=2.0): #boardDict is word2vec Tensor version of board with labels 
	sumLoss = 0 
	for key in boardDict:
		currLoss = 0
		if(key == 'blue'):
			blueTensor = boardDict[key]
			for word2vec in blueTensor:
				#print("word2vecShape: {},outputVectorShape:{}".format(word2vec.shape,outputVector.shape))
				#for every individual word2vec vector, sum cosine similarity
				currLoss += 1 - F.cosine_similarity(word2vec,outputVector,dim=0) #cosine dist = 1-cosine_similarity
			currLoss = -torch.log(torch.mean(currLoss))
		elif(key == 'red'):
			redTensor = boardDict[key]
			for word2vec in redTensor:
			 	#for every individual word2vec vector, sum cosine similarity
			 	currLoss += 1 - F.cosine_similarity(word2vec,outputVector,dim=0) #cosine dist = 1-cosine_similarity
			currLoss = torch.log(torch.mean(currLoss))
		elif(key =='assassin'):
			assassinVector = boardDict[key] #should be 1,len(word2Vec)
			currLoss = torch.log(torch.mean(assassinWeight*(1-F.cosine_similarity(assassinVector,outputVector,dim=0))))
		# print(currLoss.shape)
		sumLoss += currLoss
	return sumLoss


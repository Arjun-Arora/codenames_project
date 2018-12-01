import sys
sys.path.append("../src")
sys.path.append("../models")
import utils
import linear_model
import torch
import gensim
import numpy as np


model_path = "basicmodel.pt"
googlemodel = gensim.models.KeyedVectors.load_word2vec_format('../assets/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin.gz', binary=True, limit=10000)

#list of words
kWords = 10
clues = googlemodel.index2word[:kWords]
clues_w2v = [torch.Tensor(googlemodel.wv[word]) for word in clues]


def BasicTest(): 
	ListOfBoardDicts = utils.readBoards()
	Input = [] 
	for BoardDict in ListOfBoardDicts:
		Input.append(torch.cat(list(BoardDict.values())))
		#print(BoardDict.values())
	TensorInput = torch.stack(Input) #(NumofBoards,sizeofBoards,word2vec_size)
	# print(TensorInput.shape)
	#data input,output size
	numBoards, boardSz, w2v_sz = TensorInput.shape
	TensorInput = TensorInput.view(numBoards,-1)
	# N,D_in,D_out = TensorInput.shape
	#hidden layer size
	# H = 1000
	model = linear_model.createSequential(boardSz, w2v_sz, k=kWords)
	# print(model)
	# criterion = linear_model.BasicLoss
	criterion = linear_model.BasicLoss

	optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
	for t in range(100):#epochs
		output_vectors = model(TensorInput)
		# print("model output shape: {}".format(output_vectors.shape))
		loss = 0 
		# print("output_vectors shape: {}".format(output_vectors.shape))
		
		for i in range(output_vectors.shape[0]):
			# print(output_vectors[i,:].shape)
			output_vector = output_vectors[i,:]
			# print("output vector shape: {}".format(output_vector.shape))
			# print("output_vector shape before input:{}".format(output_vector.shape))
			loss += criterion(output_vector,ListOfBoardDicts[i], clues_w2v)
		#print out loss
		# print(t, loss.item())

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	torch.save(model, model_path)

def Inference(board):

	#unsqueezed to ensure that input is (1,ProbDistribution) because Softmax is done across the 1 dimension
	Input = torch.unsqueeze(torch.cat(list(board.values())).view(-1),dim=0)

	# print("Input shape: {}".format(Input.shape))
	# googlemodel = gensim.models.KeyedVectors.load_word2vec_format('../assets/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin', binary=True, limit=10000)
	# print (googlemodel.index2word[-50:])
	model = torch.load(model_path)
	# torch.zero_grad()
	with torch.no_grad():
		model = model.eval()

		output = model(Input)
		# print (output)
		nx = output.detach().numpy()
		# max_prob = max(nx)
		idx = np.argmax(nx)
		# print ("clue: {}".format(clues[idx]))

		criterion = linear_model.BasicLoss
		print("eval criterion: {}".format(criterion(output, board, clues_w2v))) #also, output isn't the same as clue - clue is the nearest word
		return clues[idx]
	


BasicTest() 
ListOfBoardDicts = utils.readBoards()
for b in ListOfBoardDicts:
	print(Inference(b))
	break
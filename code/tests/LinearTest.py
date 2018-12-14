import sys
sys.path.append("../src")
sys.path.append("../models")
import utils
import linear_model
import torch
import gensim
import numpy as np
from tqdm import tqdm

model_path = "basicmodel.pt"
googlemodel = gensim.models.KeyedVectors.load_word2vec_format('../assets/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin', binary=True, limit=10000)

#list of words
kWords = 1000
clues = googlemodel.index2word[:kWords]
clues_w2v = [torch.Tensor(googlemodel.wv[word]) for word in clues]

def BasicTest(): 
	ListOfBoardDicts = utils.readBoards()
	dev_boards = utils.readBoards("../assets/dev_board_list.json")
	Input = [] 
	for BoardDict in ListOfBoardDicts:
		Input.append(torch.cat(list(BoardDict.values())))
		#print(BoardDict.values())
	TensorInput = torch.stack(Input) #(NumofBoards,sizeofBoards,word2vec_size)
	# print(TensorInput.shape)
	#data input,output size
	numBoards, boardSz, w2v_sz = TensorInput.shape
	TensorInput = TensorInput.view(numBoards,-1) #flatten out: numboards, sizeofboards x word2ve sz (2d)
	model = linear_model.createSequential(boardSz, w2v_sz, k=kWords)
	# print(model)
	print ("made model")
	criterion = linear_model.BasicLoss

	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) #alt to sgd
	batchsz = 16 #consider loss of 16 boards at a time
	for t in range(100):#epochs
		print ("epoch " + str(t))
		avg_loss = 0
		for b in tqdm(range(int(numBoards/batchsz))):
			output_vectors = model(TensorInput[b:b+batchsz,:])
			# print("model output shape: {}".format(output_vectors.shape))
			loss = 0 
			# print("output_vectors shape: {}".format(output_vectors.shape))
			
			for i in range(output_vectors.shape[0]):
				# print(output_vectors[i,:].shape)
				output_vector = output_vectors[i,:]
				# print("output vector shape: {}".format(output_vector.shape))
				# print("output_vector shape before input:{}".format(output_vector.shape))
				loss += criterion(output_vector,ListOfBoardDicts[i], clues_w2v)
				avg_loss+=loss.item()
			
			#print out loss
			#print(t, loss.item())

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		
		for board in dev_boards[:5]:
			print (Inference(board, trained_model=model))
		print ("average loss for epoch " + str(avg_loss / (numBoards/batchsz)))

	torch.save(model, model_path)

def Inference(board, model_path=None, trained_model=None):

	#unsqueezed to ensure that input is (1,ProbDistribution) because Softmax is done across the 1 dimension
	Input = torch.unsqueeze(torch.cat(list(board.values())).view(-1),dim=0)
	#load saved, fully trained model from model path
	if model_path:
		model = torch.load(model_path)
	else:
		model = trained_model

	with torch.no_grad():
		model = model.eval()

		output = model(Input)
		# print (output)
		nx = output.detach().numpy()
		idx = np.argmax(nx)
		# print ("clue: {}".format(clues[idx]))

		criterion = linear_model.BasicLoss
		print("eval criterion: {}".format(criterion(output, board, clues_w2v))) #also, output isn't the same as clue - clue is the nearest word
		# board_words = utils.readBoardsWords()
		# print("board")
		# '{0:9} {0:8}'.format(board_words['blue'], board_words['red'])
		return clues[idx]
	


BasicTest() 
dev_boards = utils.readBoards("../assets/dev_board_list.json")
for board in dev_boards[:5]:
	Inference(board, model)


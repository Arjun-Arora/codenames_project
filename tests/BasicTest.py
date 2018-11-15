import sys
sys.path.append("../src")
sys.path.append("../models")
import utils
import basic_model
import torch
import gensim

model_path = "basicmodel.pt"

def BasicTest(): 
	ListOfBoardDicts = utils.readBoards()
	Input = [] 
	for BoardDict in ListOfBoardDicts:
		Input.append(torch.cat(list(BoardDict.values())))
		#print(BoardDict.values())
	TensorInput = torch.stack(Input) #(NumofBoards,sizeofBoards,word2vec_size)
	# print(TensorInput.shape)
	#data input,output size 
	N,D_in,D_out = TensorInput.shape
	#hidden layer size
	H = 1000

	model = basic_model.BasicModel(D_out,H,D_out)
	criterion = basic_model.BasicLoss
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
	for t in range(100):#epochs
		output_vectors = model(TensorInput)
		loss = 0 
		# print("output_vectors shape: {}".format(output_vectors.shape))
		
		for i in range(output_vectors.shape[0]):
			# print(output_vectors[i,:].shape)
			output_vector = output_vectors[i,:]
			# print("output_vector shape before input:{}".format(output_vector.shape))
			loss += criterion(output_vector,ListOfBoardDicts[i])
		#print out loss
		# print(t, loss.item())

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	torch.save(model, model_path)

def Inference(board):

	Input = torch.cat(list(board.values())).unsqueeze(dim=0)
	googlemodel = gensim.models.KeyedVectors.load_word2vec_format('../assets/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin', binary=True, limit=500000)

	model = torch.load(model_path)
	# torch.zero_grad()
	model = model.eval()

	output = model(Input)
	# print (googlemodel.wv.vocab)
	clue = utils.findNearestWord(list(googlemodel.wv.vocab.keys()), googlemodel, output.data.numpy())
	print(board)
	print (clue)

BasicTest()
ListOfBoardDicts = utils.readBoards()
Inference(ListOfBoardDicts[4])
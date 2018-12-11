import sys
sys.path.append("./src")
sys.path.append("./models")
import utils
from basic_model import BasicModel,BasicLoss
import torch
import gensim
from tqdm import tqdm
import torch.nn.functional as F

model_path = "basicmodel.pt"

def init_weights(m):
	for param in m.parameters():
		if len(param.shape) >= 2:
			torch.nn.init.xavier_uniform(param)

def TrainBasicModel(epochs=100,batch_size=32,save_path=None):
	ListOfBoardDicts = utils.readBoards()
	Input = []

	for BoardDict in ListOfBoardDicts:
		Input.append(torch.cat(list(BoardDict.values())))
		#print(BoardDict.values())

	TensorInput = torch.stack(Input) #(NumofBoards,sizeofBoards,word2vec_size)
	# print(TensorInput.shape)

	#data input,output size
	numBoards, boardSz, w2v_sz = TensorInput.shape
	TensorInput = TensorInput.view(numBoards, -1)
	print("number of boards in training set: {}".format(numBoards))
	#hidden layer size
	h = 200

	#dtype
	dtype = torch.FloatTensor

	model = BasicModel(boardSz, w2v_sz,h=200)

	if torch.cuda.is_available():
		model = model.cuda()
		dtype = torch.cuda.FloatTensor
		print ('GPU: {}'.format(torch.cuda.get_device_name(0)))

	model.apply(init_weights)

	criterion = BasicLoss
	# criterion = basic_model.KWordLoss

	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,weight_decay=1e0)
	for t in range(1,epochs+1):#epochs
		avg_loss = 0
		optimizer.zero_grad()
		num_iters = int(len(TensorInput)/batch_size)

		for i in tqdm(range(num_iters)):
			output_vectors = model(TensorInput[i:i+batch_size,:])
			loss = 0
			# print("output_vectors shape: {}".format(output_vectors.shape))
			for i in range(output_vectors.shape[0]):
				# print(output_vectors[i,:].shape)
				output_vector = output_vectors[i,:]
				# print("output_vector shape before input:{}".format(output_vector.shape))
				loss += criterion(output_vector,ListOfBoardDicts[i])
			avg_loss += loss.item()/output_vectors.shape[0]
				# print("loss per iter: {}".format(loss.item()))
			#print out loss
			loss.backward()
			optimizer.step()
		print("avg loss for epoch {}: {}".format(t, avg_loss/num_iters))
	torch.save(model, model_path)

def Inference(board):
	Input = torch.cat(list(board.values())).unsqueeze(dim=0)
	numBoards, boardSz, w2v_sz = Input.shape
	googlemodel = gensim.models.KeyedVectors.load_word2vec_format('./assets/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin', binary=True, limit=10000)
	# print (googlemodel.index2word[-50:])
	model = torch.load(model_path)
	with torch.no_grad():
		model = model.eval()
		# print("input.shape: {}".format(Input.shape))
		Input = Input.view(numBoards,-1)
		output = model(Input)
		# print (googlemodel.wv.vocab)
		clue = utils.findNearestWord(list(googlemodel.wv.vocab.keys()), googlemodel, output.squeeze(dim=0).data.numpy())

		# print("shape of output:{} shape of clue: {} ".format(output.shape,googlemodel.wv[clue].shape))
		print("cosine similarity of output and clue: {}".format(F.cosine_similarity(output.squeeze(dim=0),torch.FloatTensor(googlemodel.wv[clue]),dim=0)))
		# print(output)
		# tabledata = list[output.data.numpy().tolist()]
		# print(board)
		# print ("clue: {}".format(clue))
		criterion = BasicLoss
		# print (criterion(output, board)) #also, output isn't the same as clue - clue is the nearest word
		# print (criterion(torch.Tensor(googlemodel.wv[clue]), board))
		return clue


TrainBasicModel(epochs=30)
ListOfBoardDicts = utils.readBoards("./assets/dev_board_list.json")
ListOfBoardDictsWords = utils.readWordBoards("./assets/dev_board_list.json")
for i in range(len(ListOfBoardDicts[:5])):
	b = ListOfBoardDicts[i]
	b_words = ListOfBoardDictsWords[i]
	print ("clue: {}".format(Inference(b)))
	print ("board: {}".format(b_words))
	# print ("board: {}".format(b))
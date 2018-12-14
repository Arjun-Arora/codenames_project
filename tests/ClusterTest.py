import sys
sys.path.append("models")
from clustering_model import codenamesCluster, basicLoss, basicCentroid
sys.path.append("src")
import glove_utils as glove 
from random import shuffle



embeddings = glove.load_glove_model()
# print embeddings.keys()[:100]

boards = glove.get_boards()
idx = 1
for board in boards:
	print("Experiment: {}".format(idx))
	vals = board.values()
	# print(vals)
	board_vals = []
	for val in vals:
		for v in val:
			board_vals.append(v)
	shuffle(board_vals)
	for i in range(5):
  		print('\t\t'.join(board_vals[i*5:i*5+5]))
	print("\n")
	board_words = []
	for key in board:
		board_words += board[key]
	min_loss, best_clue,match = codenamesCluster(board, embeddings, basicCentroid, basicLoss, b=3)
	
	print ("clue:{} ,{}".format(glove.find_nearest_word(embeddings, best_clue, board_words),len(match)))
	input("Press Enter to continue...")
	print (board)
	print(match)
	print("\n")
	idx += 1

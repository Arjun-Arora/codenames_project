import sys
sys.path.append("models")
from clustering_model import codenamesCluster, basicLoss, basicCentroid, inBetweenVector
sys.path.append("src")
import glove_utils as glove 


embeddings = glove.load_glove_model()
# print embeddings.keys()[:100]

boards = glove.get_boards()

for board in boards:
	print (board)
	board_words = []
	for key in board:
		board_words += board[key]
	min_loss, best_clue = codenamesCluster(board, embeddings, basicCentroid, basicLoss, 2)
	
	print (glove.find_nearest_word(embeddings, best_clue, board_words))
	print ("\n")
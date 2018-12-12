import sys
sys.path.append("models")
from clustering_model import codenamesCluster, basicLoss, basicCentroid
sys.path.append("src")
import glove_utils as glove 


embeddings = glove.load_glove_model()
boards = glove.get_boards()

for board in boards:
	print (board)
	board_words = []
	for key in board:
		# print key
		board_words += board[key]
	# board_list = board['blue'] + board[u'red'] + board[u'assassin'] + blue[u'neutral']
	min_loss, best_clue = codenamesCluster(board, embeddings, basicCentroid, basicLoss)
	# print best_clue, min_loss
	
	print (glove.find_nearest_word(embeddings, best_clue, board_words))
	# break
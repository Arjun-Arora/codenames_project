import sys
sys.path.append("models")
from clustering_model import codenamesCluster, basicLoss, basicCentroid
sys.path.append("src")
import glove_utils as glove 


embeddings = glove.load_glove_model()
boards = glove.get_boards()

for board in boards:
	min_loss, best_clue = codenamesCluster(board, embeddings, basicCentroid, basicLoss)
	print board
	print best_clue, min_loss
	print glove.find_nearest_word(embeddings, best_clue)
	# break
import sys
sys.path.append("models")
from clustering_model import basicLoss, basicCentroid
sys.path.append("src")
import glove_utils as glove 
from collections import defaultdict


embeddings = glove.load_glove_model()
boards = glove.get_boards()

for board in boards:
	print (board)
	board_words = []
	for key in board:
		board_words += board[key]

	words_to_avoid = [str(word) for word in board_words]
	embeddingBoard = defaultdict(list)
	for i, (team,wordList) in enumerate(board.items()):
		for word in wordList:
			embeddingBoard[team].append(embeddings[word])

	best_clue = basicCentroid(embeddingBoard["blue"])
	
	print (glove.find_nearest_word(embeddings, best_clue, board_words))
	# break
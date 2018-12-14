import random
import copy
import json
import sys

sys.path.append("../assets")


nboards = 1000

blue_len = 9
red_len = 8
assassin_len = 1
neutral_len = 7
board_len = 25

files_gensim = ["assets/gensim_dev_board_list.json", "assets/gensim_board_list.json"]
files_glove = ["assets/dev_board_list.json", "assets/board_list.json"]


words = []
f = open("assets/gensim_word_list.txt", "r")
words = [line for line in f.read().split()]

for file in files_gensim:
	f2 = open(file, "w+")
	for i in range(nboards):
		board_dic = {}
		board = random.sample(words, board_len)
		board_dic['blue'] = board[ : blue_len]
		board_dic['red'] = board[blue_len : blue_len+red_len]
		board_dic['assassin'] = board[blue_len+red_len : blue_len+red_len+assassin_len]
		board_dic['neutral'] = board[blue_len+red_len+assassin_len:blue_len+red_len+assassin_len + neutral_len] #for now, make the neutrals = red for simplicity

		
		json.dump(board_dic, f2)
		f2.write("\n")

words = []
f = open("assets/glove_list.txt", "r")
words = [line for line in f.read().split()]
for file in files_glove:
	f2 = open(file, "w+")
	for i in range(nboards):
		board_dic = {}
		board = random.sample(words, board_len)
		board_dic['blue'] = board[ : blue_len]
		board_dic['red'] = board[blue_len : blue_len+red_len]
		board_dic['assassin'] = board[blue_len+red_len : blue_len+red_len+assassin_len]
		board_dic['neutral'] = board[blue_len+red_len+assassin_len:blue_len+red_len+assassin_len + neutral_len] #for now, make the neutrals = red for simplicity

		
		json.dump(board_dic, f2)
		f2.write("\n")

# f2 = open("assets/dev_board_list.json", "w")
# for i in range(nboards):
# 	board_dic = {}
# 	board = random.sample(words, board_len)
# 	board_dic['blue'] = board[ : blue_len]
# 	board_dic['red'] = board[blue_len : blue_len+red_len]
# 	board_dic['assassin'] = board[blue_len+red_len : blue_len+red_len+assassin_len]
# 	board_dic['neutral'] = board[blue_len+red_len+assassin_len:blue_len+red_len+assassin_len + neutral_len] #for now, make the neutrals = red for simplicity


# 	json.dump(board_dic, f2)
# 	f2.write("\n")


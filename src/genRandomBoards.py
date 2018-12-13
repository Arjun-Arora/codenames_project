import random
import copy
import json
import sys
# import argparse

# parser = argparse.ArgumentParser(description='generate graph based on kNN')
# parser.add_argument('--numboards', '-k', help='num boards to generate', type=int, default=5)
# args = parser.parse_args()
# nboards = args.numboards
sys.path.append("../assets")

words = []
f = open("assets/glove_list.txt", "r")
words = [line for line in f.read().split()]

full_list = copy.deepcopy(words)
nboards = 1000

blue_len = 9
red_len = 8
assassin_len = 1
neutral_len = 7
board_len = 25
# blue_len = 1
# red_len = 1
# assassin_len = 0
# neutral_len = 0
# board_len = 2
f2 = open("assets/board_list.json", "w")
for i in range(nboards):
	board_dic = {}
	board = random.sample(words, board_len)
	board_dic['blue'] = board[ : blue_len]
	board_dic['red'] = board[blue_len : blue_len+red_len]
	board_dic['assassin'] = board[blue_len+red_len : blue_len+red_len+assassin_len]
	board_dic['neutral'] = board[blue_len+red_len+assassin_len:blue_len+red_len+assassin_len + neutral_len] #for now, make the neutrals = red for simplicity


	json.dump(board_dic, f2)
	f2.write("\n")


f2 = open("assets/dev_board_list.json", "w")
for i in range(nboards):
	board_dic = {}
	board = random.sample(words, board_len)
	board_dic['blue'] = board[ : blue_len]
	board_dic['red'] = board[blue_len : blue_len+red_len]
	board_dic['assassin'] = board[blue_len+red_len : blue_len+red_len+assassin_len]
	board_dic['neutral'] = board[blue_len+red_len+assassin_len:blue_len+red_len+assassin_len + neutral_len] #for now, make the neutrals = red for simplicity


	json.dump(board_dic, f2)
	f2.write("\n")


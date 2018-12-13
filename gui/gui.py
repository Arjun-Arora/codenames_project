from itertools import chain
# import sys
import scipy as sp
from scipy import spatial
import copy
import itertools
from collections import defaultdict
import numpy as np
import json
import time
# sys.path.append(".")
# sys.path.append("..")
# sys.path.append("../models")
# sys.path.append("../src")
# print(sys.path)
# from clustering_model import codenamesCluster, basicLoss, basicCentroid
# import glove_utils as glove
    
class SolarSystem:
    planets = [list (chain (planet, (index + 1,))) for index, planet in enumerate ((
        ('Mercury', 'hot', 2240),
        ('Venus', 'sulphurous', 6052),
        ('Earth', 'fertile', 6378),
        ('Mars', 'reddish', 3397),
        ('Jupiter', 'stormy', 71492),
        ('Saturn', 'ringed', 60268),
        ('Uranus', 'cold', 25559),
        ('Neptune', 'very cold', 24766) 
    ))]
    
    lines = (
        '{} is a {} planet',
        'The radius of {} is {} km',
        '{} is planet nr. {} counting from the sun'
    )
    
    def __init__ (self):
        self.lineIndex = 0
    
    def greet (self):
        self.planet = self.planets [int (Math.random () * len (self.planets))]
        document.getElementById ('greet') .innerHTML = 'Hello {}'.format (self.planet [0])
        self.explain ()
        
    def explain (self):
        document.getElementById ('explain').innerHTML = (
            self.lines [self.lineIndex] .format (self.planet [0], self.planet [self.lineIndex + 1])
        )
        self.lineIndex = (self.lineIndex + 1) % 3

    def readHint (self):
        document.getElementById ('explain').innerHTML = document.getElementById ('hint').value

class SpymasterCluster:
    def __init__ (self):
        self.boardIndex = 0
        self.embeddings = self.load_glove_model("../assets/glove.6B.50d.txt", "../assets/word_list.txt")
        self.boards = self.get_boards("../assets/dev_board_list.json")

        self.makeGuess()

    def load_glove_model(self, glove_file="assets/glove.6B.50d.txt", word_list_file="assets/word_list.txt"):
        # print "Loading Glove Model"
        f = open(glove_file,'r', encoding='utf8')
        model = {}
        w =  open(word_list_file, 'r')
        ignore_n = 50
        for idx, line in enumerate(f):
            if idx < ignore_n:
                continue
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        # print "Done.",len(model)," words loaded!"
        return model


    def get_boards(self, glove_file="assets/dev_board_list.json"):
        file = glove_file
        with open(file, 'r') as f:
            dicts = []
            for line in f:
                board = json.loads(line)
                dicts.append(board)
        return dicts

    def find_nearest_word(self, corpus, vector, words_to_avoid): 
        '''
        find nearest word function: 
        inputs:
            corpus: List of words that the corpus we're using for this codenames game
            model: this is the gensim word2vec model we're using for this function
            vector: numpy array representing a chosen word2vecVector 

        Output: String word that is nearest to word2vecVector from corpus (cosine distance)
        '''
        start = time.time()
        words_to_avoid = [str(word) for word in words_to_avoid]
        # print corpus.keys()[:10]
        min_word = ""
        min_dist = float('Inf')
        for word in corpus:
            if word not in words_to_avoid:
                curr_dist = sp.spatial.distance.cosine(corpus[word], vector)
                if curr_dist < min_dist:
                    min_word = word
                    min_dist = curr_dist
        end = time.time()
        print("Search took: {:.2f} seconds".format(end-start))
        # min_idx = sp.argmin([sp.spatial.distance.cosine(corpus[word],vector) for word in corpus if word not in words_to_avoid]) #minimize across words not across vector length
        # min_word = corpus.keys()[min_idx]
        return min_word

    def basicLoss(self, blue_list, red_list=None, assassin_list=None):
        dist = 0
        blue_norm = len(blue_list)
        red_norm = len(red_list)
        for blue_pair in itertools.combinations(blue_list, 2): #for every pair of words in this list
            #dist +=  spatial.distance.cosine(blue_pair[0], blue_pair[1]) / blue_norm
            curr_dist = spatial.distance.cosine(blue_pair[0], blue_pair[1]) / blue_norm
            for red_word in red_list:

                red_blue0 = spatial.distance.cosine(red_word, blue_pair[0])
                red_blue1 = spatial.distance.cosine(red_word, blue_pair[1])
                curr_dist -= (red_blue0 + red_blue1) / 2 / red_norm
                # if min(red_blue, red_blue1) <= 1.5 * curr_dist:
                #     curr_dist *= 2
            dist += curr_dist

        return dist

    def basicCentroid(self, word_vecs):
        return np.sum(word_vecs, axis=0)/len(word_vecs)

    def codenamesCluster(self, codenamesBoard,embedding,centroid_fn,loss_fn,b=2,r=0,a=0):
        """
        :param codenamesBoard: dictionary of codewords (just 25 words)
        :param embedding: dictionary that goes from word -> embedding
        :param centroid_fn: function that gives most 'central' word given a list of blue vectors
        :param loss_fn: takes in 3 lists corresponding to combinations of blue, red, and assassin words vectors
        :return: best cluster center average
        """
        
        embeddingBoard = defaultdict(list)
        for i, (team,wordList) in enumerate(codenamesBoard.items()):
            for word in wordList:
                embeddingBoard[team].append(embedding[word])
        minLoss = float('Inf')
        bestClue = None

        bestCombo = None
        for blue_idxs in itertools.combinations(range(len(embeddingBoard['blue'])), b):
            blueCombinations = [embeddingBoard['blue'][idx] for idx in blue_idxs]
        # for blueCombinations in itertools.combinations(embeddingBoard['blue'], b):
            # for redCombinations in itertools.combinations(embeddingBoard['red'],r):
            curr_loss = loss_fn(blueCombinations, embeddingBoard['red']) #,redCombinations,assassinCombinations)
            if curr_loss < minLoss:
                minLoss=curr_loss
                bestClue = centroid_fn(blueCombinations)
                bestCombo = blue_idxs

        print ([codenamesBoard['blue'][idx] for idx in bestCombo])
        
        return minLoss,bestClue

    def makeGuess(self):
        board = self.boards[self.boardIndex]
        # document.getElementById ('board').innerHTML = board
        board_words = []
        for key in board:
            board_words += board[key]
        min_loss, best_clue = self.codenamesCluster(board, self.embeddings, self.basicCentroid, self.basicLoss, 2)
        
        hint = (self.find_nearest_word(self.embeddings, best_clue, board_words))
        # document.getElementById ('guess').innerHTML = hint
        
solarSystem = SolarSystem ()
spymaster = SpymasterCluster ()
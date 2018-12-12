import scipy as sp
from scipy import spatial
import copy
import itertools
from collections import defaultdict
import time
import numpy as np


def basicLoss(blue_list, red_list=None, assassin_list=None):
    dist = 0
    for blue_pair in itertools.combinations(blue_list, 2):
        # dist = 1
        dist += spatial.distance.cosine(blue_pair[0], blue_pair[1])
    return dist

def basicCentroid(word_vecs):
    return np.sum(word_vecs, axis=0)/len(word_vecs)

def codenamesCluster(codenamesBoard,embedding,centroid_fn,loss_fn,b=2,r=0,a=0):
    """
    :param codenamesBoard: dictionary of codewords (just 25 words)
    :param embedding: dictionary that goes from word -> embedding
    :param centroid_fn: function that gives most 'central' word given a list of blue vectors
    :param loss_fn: takes in 3 lists corresponding to combinations of blue, red, and assassin words vectors
    :return: best cluster center average
    """
    start = time.time()
    embeddingBoard = defaultdict(list)
    for i, (team,wordList) in enumerate(codenamesBoard.items()):
        for word in wordList:
            embeddingBoard[team].append(embedding[word])
    minLoss = float('Inf')
    bestClue = None

    for blueCombinations in itertools.combinations(embeddingBoard['blue'], b):
        # for redCombinations in itertools.combinations(embeddingBoard['red'],r):
            # for assassinCombinations in itertools.combinations(embeddingBoard['assassin'],a):
        curr_loss = loss_fn(blueCombinations) #,redCombinations,assassinCombinations)
        if curr_loss < minLoss:
            minLoss=curr_loss
            bestClue = centroid_fn(blueCombinations)
            # print blueCombinations
            # print bestClue


    end = time.time()
    print("Clustering took: {:.2f} seconds".format(start-end))
    return minLoss,bestClue




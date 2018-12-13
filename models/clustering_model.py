import scipy as sp
from scipy import spatial
import copy
import itertools
from collections import defaultdict
import numpy as np

# def inBetweenVector(x,y):
#     theta = sp.arccos(x.dot(y)/(sp.linalg.norm(x)*sp.linalg.norm(y)))
#     angle = theta/2.0
# #     print(angle)
#     N = x.shape[0]
#     rotation_matrix = sp.diag(sp.ones(N))
#     # print(rotation_matrix)
#     rotation_matrix[0,0] = sp.cos(angle)
#     rotation_matrix[1,1] = sp.cos(angle)
#     rotation_matrix[0,1] = -sp.sin(angle)
#     rotation_matrix[1,0] = sp.sin(angle)
#     return rotation_matrix.dot(x)


def basicLoss(blue_list, red_list=None, assassin=None):
    dist = 0
    blue_norm = len(blue_list)
    red_norm = len(red_list)
    # assassin_weight = 
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
    print "new!"
    
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




####
Codenames Project - CS 221 Fall 2018
####

By 
Heidi Chen
Arjun Arora
Daniel Classon

DATA
data.zip contains all assets for this project. Before running any scripts, unzip data into the root project folder, and rename it as “assets”. 
-- gensim_word_list.txt: all words used in Codenames, formatted to match gensim’s Google News corpus.
-- glove_list.txt: all words used in Codenames, formatted to match GloVe’s Wikipedia word corpus. 
-- missing / need to download separately: http://nlp.stanford.edu/data/glove.6B.zip (Wikipedia, 50d). Pre-trained GloVe vectors.
-- missing / need to download separately: https://github.com/mmihaltz/word2vec-GoogleNews-vectors. Pre-trained Gensim vectors.
-- all *board_list.json’s: training data, randomly generated. Can be generated again from word_lists.

CODE
code.zip contains all models and scripts. 

/baseline/codenames_blog: baseline jupyter notebook forked from [3].
/models/basic_model.py: Neural net with basic and k-word loss.
/models/clustering_model.py: Clustering model with variable k input.
/models/linear_model.py: [Not runnable, cannibalized to basic_model] linear neural net.

/src/utils.py: util functions for gensim and neural net.
/src/glove_utils.py: util functions for clustering and glove.

All files in /tests run their corresponding models.

RUNNING SCRIPTS
To run the neural net, activate Python 3 and Pytorch installed. Then, run 
> python tests/BasicTest.py

To run clustering, activate Python 2 and run
> python tests/ClusterTest.py

To run basic exhaustive, activate Python 2 and run
> python tests/ExhaustiveTest.py

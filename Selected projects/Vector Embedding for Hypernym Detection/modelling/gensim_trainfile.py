import pre_processing
import file_readers
import distr_matrix
import batch_creator
import dive_net
import scipy
import numpy as np
import os
import math
import time

from gensim.models import Word2Vec

if __name__ == '__main__':
    #get preprocessed text and vocab list
    with open('wackypedia_512k_POS0_wostop', 'r') as f:
        text_pre = f.read()
    text_pre_list = text_pre.split('\n')
    sentences = [k.split(' ') for k in text_pre_list]
    model = Word2Vec(sentences, size=100, window=10, sg=1, min_count=5, workers=2, batch_words=128)
    model.save('gensim_w2v')

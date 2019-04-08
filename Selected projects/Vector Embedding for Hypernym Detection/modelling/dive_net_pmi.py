import pre_processing
import dive_net
from scipy import sparse
import numpy as np
import os
import math
import time
import pickle
# pmi filtering, without w2v, with negative clipping
import tensorflow as tf
   
if __name__ == '__main__':
    #get preprocessed text and vocab list
    text_pre_path = './data/pre_processed_text.txt'
    vocab_path = os.getcwd()+'/data/SemEval18-Task9_Training/SemEval18-Task9/vocabulary/2B.music.vocabulary.txt'
    training_path = os.getcwd()+'/data/SemEval18-Task9_Training/SemEval18-Task9/training/data/2B.music.training.data.txt'
    
    #initialize 
    start = time.time()
    #matrix, pmi_mat, index, text_pre = generate_pmi_context(vocab_path, training_path, text_pre_path, verbose=True)
    #np.save('context_matrix', matrix)
    #np.save('pmi_matrix', pmi_mat)
    #import json
    #with open('word_index.json', 'w') as f:
    #    json.dump(index, f)
    fileName = 'wackypedia_512k_POS0_wostop'
    # fileName = 'test'
    window = 10
    batch_size = 128
    tokens = pre_processing.getTokens(fileName)
    dictionary = pre_processing.makeDictionary(tokens)
    ctxMatrix = pre_processing.getCtxMatrix(tokens, dictionary, window)
    print('finished building context matrix',ctxMatrix.shape)
    pmiMatrix = pre_processing.getPmiMatrix2(ctxMatrix)
    pickle.dump(tokens,open("tokens","wb"))
    pickle.dump(dictionary, open("dictionary","wb"))
    print('converting ctxMatrix')
    compressed_ctx = sparse.lil_matrix.tocsr(ctxMatrix)
    print('conversion complete')
    #compressed_pmi = sparse.lil_matrix.tocsr(pmiMatrix)
    print('storing ctxMatrix')
    sparse.save_npz('ctxmatrix', compressed_ctx)
    print('storing pmiMatrix')
    sparse.save_npz('pmimatrix', pmiMatrix)
    print('matrix and pmi computed and saved, consuming:',time.time()-start)
    # tokens = pickle.load(open("tokens","rb"))
    # dictionary = pickle.load(open("dictionary","rb"))
    # ctxMatrix = pickle.load(open("ctxmatrix","rb"))
    # pmiMatrix = pickle.load(open("pmimatrix","rb"))
    dnet = dive_net.Dive_net(dictionary, embedding_size=100, batch_size=128, neg_samples=2) #initialize net
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        dnet.sess = sess
        print('batchCreator')
        batches = pre_processing.batchCreator(fileName, window, batch_size, dictionary, pmiMatrix)
        print('epoch {}, batches generated')
        for i in range(15): #15 epochs in paper
            for batch in batches:
                print('epoch {}, generating batches'.format(i))
                x =  np.resize(batch[0], batch_size).reshape(batch_size, 1)
                y =  np.resize(batch[1], batch_size).reshape(batch_size, 1)
                x = x.flatten()
                # y = y.flatten()
                cur_loss = dnet.train_batch(x, y)
                print('end of epoch loss: {}'.format(cur_loss))
        saver = tf.train.Saver()
        saver.save(sess, "./pmi_dnet")

    
    
    
    
    
    
    

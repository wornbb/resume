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

def generate_pmi_context(vocab_path, training_path, text_pre_path, verbose=False):
    '''
    Given paths to vocabulary, training hyponyms and pre-processed text, returns a PMI matrix, a distributional matrix and an index mapping word strings to integers
    '''
    text_pre = file_readers.read_file(text_pre_path)
    vocab_list = file_readers.read_vocab_list(vocab_path) + [k[0] for k in file_readers.read_hyponyms(training_path)]
    
    #get global context matrix and pmi (this could take a while)
    if(verbose):
        print('starting context matrix building')
        start = time.time()
    matrix, index = distr_matrix.create_context_matrix(text_pre, num_words=0, window=10, word_list=vocab_list)
    np.save('context_matrix', matrix)
    import json
    with open('index.json', 'w') as f:
        json.dump(index, f)

    if(verbose):
        print('context matrix took: {}'.format(time.time()-start))
        start = time.time()
    print('started PMI matrix building')
    pmi_mat = distr_matrix.pmi_matrix(matrix, inplace=False)
    
    if(verbose):
        print('pmi matrix took: {}'.format(time.time()-start))
    np.save('pmi_matrix', pmi_mat)
    
    
    return matrix, pmi_mat, index, text_pre
   
if __name__ == '__main__':
    #get preprocessed text and vocab list
    text_pre_path = './data/pre_processed_text.txt'
    vocab_path = os.getcwd()+'/data/SemEval18-Task9_Training/SemEval18-Task9/vocabulary/2B.music.vocabulary.txt'
    training_path = os.getcwd()+'/data/SemEval18-Task9_Training/SemEval18-Task9/training/data/2B.music.training.data.txt'
    
    #initialize 
    start = time.time()
    matrix, pmi_mat, index, text_pre = generate_pmi_context(vocab_path, training_path, text_pre_path, verbose=True)
    #np.save('context_matrix', matrix)
    #np.save('pmi_matrix', pmi_mat)
    #import json
    #with open('word_index.json', 'w') as f:
    #    json.dump(index, f)
        
    print('matrix and pmi computed and saved, consuming:',time.time()-start)
    
    dnet = dive_net.Dive_net(index, embedding_size=100, batch_size=128, neg_samples=2) #initialize net
    
    for i in range(15): #15 epochs in paper
        print('epoch {}, generating batches'.format(i))
        batches = batch_creator.create_batches(text_pre, pmi_mat, index, window=10, batch_size=128)
        print('epoch {}, batches generated')
        for batch in batches:
            cur_loss = dnet.train_batch(batch[0], batch[1])
        print('end of epoch loss: {}'.format(cur_loss))
    
    
    
    
    
    
    
    
    

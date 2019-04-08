#import batch_creator
#tensorflow basic code for skip gram model.
import tensorflow as tf
import math 
from keras.constraints import Constraint
from keras import backend as K
import numpy as np
import dill
class Dive_net():
    '''
    Class to implement the DIVE neural network to recover self.embeddings. Trained like a classic skip_gram architecture, with 2 important differences to maintain hierarchical relationships between word vectors: (1) we use a clipping operation to keep all self.embeddings above 0, and (2) we use uniform sampling from all possible contexts, not the default Zipfian sampling which samples more common words more frequently.
    '''
    def __init__(self, word_index, embedding_size=3, batch_size=10, neg_samples=5, w2v=False):
        '''
        Initializes the network. 
        
        Args:
        word_index - a dict which maps word strings to word_id integers
        embedding size - an int, the number of neurons in the hidden layer
        batch_size - int, the number of positive samples fed in each batch
        neg_samples - int, the number of negative samples per batch (can't be more than vocabulary_size).
        w2v - bool, if true we don't use PPMI sampling
        '''
        
        self.word_index = word_index
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.neg_samples = neg_samples
        self.w2v = w2v
        self.init_graph()
        
    def init_graph(self):
        '''
        Initializes the graph with the parameters set above. Initializes the session for this graph.

        '''
        #hyperparameters
        vocabulary_size = len(self.word_index)

        #actual self.embeddings
        self.embeddings = tf.Variable(tf.random_uniform([vocabulary_size, self.embedding_size], -1.0, 1.0), name='embeddings')
        self.clip_op = tf.assign(self.embeddings, tf.clip_by_value(self.embeddings, 0, np.infty))

        #weights and biases going from embedding layer to output layer of |V| nodes.
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, self.embedding_size],
                                                     stddev=1.0 / math.sqrt(self.embedding_size)), name='nce_weights')
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]), name='biases')

        # Placeholders for inputs
        self.batch_x = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.batch_y = tf.placeholder(tf.int64, shape=[self.batch_size, 1])

        #look up the embedding vector for each of source words in the batch (for which we must predict context distribution)
        embed = tf.nn.embedding_lookup(self.embeddings, self.batch_x)
  
        # Compute the NCE loss, (using a sample of the negative labels each time)
        #note the candidate sampler is UNIFORM from the negative distribution - this is critical, and different from the default.
                
        candidate_sampler = tf.nn.uniform_candidate_sampler(
                                    true_classes = self.batch_y,
                                    num_true = 1,
                                    num_sampled = self.neg_samples,
                                    unique = True,
                                    range_max = vocabulary_size)
        

        if(self.w2v):
            #w2v
            self.loss = tf.reduce_mean(
          tf.nn.nce_loss(weights=nce_weights,
                         biases=nce_biases,
                         labels=self.batch_y,
                         inputs=embed,
                         num_sampled=self.neg_samples,
                         num_classes=vocabulary_size,
                        remove_accidental_hits=True))   
        else:
            #dive
            self.loss = tf.reduce_mean(
          tf.nn.nce_loss(weights=nce_weights,
                         biases=nce_biases,
                         labels=self.batch_y,
                         inputs=embed,
                         num_sampled=self.neg_samples,
                         num_classes=vocabulary_size,
                         sampled_values=candidate_sampler,
                        remove_accidental_hits=True))
        
         
        # We use stochastic gd 
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(self.loss)
        
        #initialize the graph.
        # self.sess = tf.Session()
        # init = tf.global_variables_initializer()
        # self.sess.run(init)


    def train_batch(self, inputs, labels, clipping=True):
        '''
        Given inputs and labels - runs one iteration of network, updates self.embeddings/weights, and does a clipping operation to remove negative weights. Returns the current loss (using logit and uniform negative sampling, as outlined in DIVE paper).
        
        Inputs:
        :inputs, labels: - numpy arrays of shapes (batch_size,) and (batch_size,1), representing contexts and their target words, mapped to integers using the same index dictionary used to initalize the network.
        
        Output:
        :cur_loss: - the current (nce) loss.
        '''
        feed_dict = {self.batch_x: inputs, self.batch_y:labels}
        _, cur_loss = self.sess.run([self.optimizer, self.loss], feed_dict = feed_dict)
        if(clipping):
            self.sess.run(self.clip_op) #remove negative weights
        return cur_loss
              
              
              
            

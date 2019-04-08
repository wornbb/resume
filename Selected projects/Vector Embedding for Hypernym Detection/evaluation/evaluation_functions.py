import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scipy
import math
import time
import urllib.request
import pandas as pd

def weeds_precision(hyperv, hypov):
    '''
    Inclusion metric: returns the weeds precision metric for these two columns of a context matrix
    
    Input:
    hyperv - a hypernym word vector from a row in the distributional matrix
    hypov - a hyponym word vector from a row in the distributional matrix
    
    Output: a float in [0,1]
    '''

    sum_match = 0
    sum_hypo = 0
    
    #iterate through, if both are non null. This can (SHOULD) be optimized more for sparse matrices.
    for i in range(len(hyperv)):
        if((hyperv[i]>0) and (hypov[i]>0)):
            sum_match += hypov[i]
            sum_hypo += hypov[i]
        elif(hypov[i]>0):
            sum_hypo += hypov[i]
    
    return sum_match/sum_hypo

def cde(hyperv, hypov):
    '''
    Inclusion metric: returns the cde measure for inclusion for these two word embeddings
    
    Input:
    hyperv - a hypernym word vector from a row in the distributional matrix
    hypov - a hyponym word vector from a row in the distributional matrix
    
    Output: a float in [0,1]
    
    '''
    min_overlaps = np.minimum(hyperv, hypov).sum()
    sum_hypo = hypo_vector.sum()
    
    return min_overlaps/sum_hypo

def invCL(hyperv, hypov):
    '''
    Inclusion metric: returns the inverse CL (similar to CDE) for these two word embeddings
    
    Input:
    hyperv - a hypernym word vector from a row in the distributional matrix
    hypov - a hyponym word vector from a row in the distributional matrix
    
    Output: an integer in [0,1]
    '''
    try:
        icl = (cde(hyperv, hypov)*(1-cde(hyperv, hypov)))**0.5
    except:
        return 0.0
    
    return icl

def cos_sim(v1, v2, matrix=None):
    from sklearn.metrics.pairwise import cosine_similarity

    '''
    Similarity metric: Given any two vectors (can be DIVE or entries of dist matrix), calculates the cosine similarity and returns it as a float.
    If matrix is not None, computes cosine similarities of all entries in the table and returns a matrix.
    '''
    if(matrix is not None):
        sims = cosine_similarity(matrix)
        return sims
    else:
        sim = cosine_similarity(v1, v2)
        return sim[0,0]

def top_context_entropies(context_matrix, pmi_matrix, top_c=100):
    '''
    Given a matrix and its pointwise mutual information matrix, returns the top context entropies for each row, useful for SQLS metrics.
    Returns a dictionary of the median entropy of each row of the context matrix.
    
    Input:
    top_c - the number of top contexts to use. We sort by PPMI.
    pmi_matrix - a positive pmi matrix computer previously
    context_matrix - some sort of semantic matrix, stored as a lil_matrix (sparse bag of words)
    
    Output:
    mean_entropy_dict - the average entropy of the top contexts of this word. 
    '''

    start = time.time()
    top_pmi_dict = {}
    top_contexts_set = set([])
    
    #get top contexts for each word, save in dict
    for i in range(context_matrix.shape[0]):
        sorted_pmi = np.argsort(-1*pmi_matrix[i,:].toarray().reshape(-1), axis=0)
        top_pmis = sorted_pmi[:top_c]
        top_pmi_dict[i] = top_pmis
        top_contexts_set = top_contexts_set.union(set(top_pmis))
        
    #for each of these contexts, get entropy. Transpose the matrix since it's faster to iterate over rows.
    context_matrix_t = context_matrix.transpose()
    entropy_vals = {}
    for k in top_contexts_set:
        row_nz = context_matrix_t.getrow(k).nonzero()
        row = context_matrix_t.getrow(k)[row_nz].toarray()[0]
        row_sum = row.sum()
        entropy_row = -sum([((entry/row_sum) * math.log((entry/row_sum),2)) for entry in row])
        entropy_vals[k] = entropy_row
    
    #now have entropy of all contexts
    mean_entropy_dict = {}
    for k in top_pmi_dict.keys():
        mean_entropy_dict[k] = np.mean([entropy_vals[z] for z in top_pmi_dict[k]])
        
    print('sqls row took:', time.time()-start)
    return mean_entropy_dict

def top_target_entropies(context_matrix):
    '''
    Given a context matrix, computes the entropies for each row and returns this as a dict.
    
    Input:
    context_matrix - a lil_matrix context matrix, rows are target words
    
    Output:
    target_ents - a dict with entropies for all rows of matrix
    '''
    target_ents = {}
    
    for k in range(context_matrix.shape[0]):
        row_nz = context_matrix.getrow(k).nonzero()
        row = context_matrix.getrow(k)[row_nz].toarray()[0]
        row_sum = row.sum()
        entropy_row = -sum([((entry/row_sum) * math.log((entry/row_sum),2)) for entry in row])
        target_ents[k] = entropy_row
    
    return target_ents

def sqls_sub(edict, hyper, hypo):
    '''
    Generality metric: given a dictionary of top context entropy values, and 2 indices (integers) of hyper and hyponyms, returns the SQLS sub metric.
    Input:
    edict - a dictionary mapping word integers to the mean entropy of their top contexts
    hyper/hypo - integer values corresponding to the indices of the words we're working with.
    
    Output:
    sqls_sub - an integer.
    '''
    return edict[hyper] - edict[hypo]

def sqls_row(target_edict, hyper, hypo):
    '''
    Generality metric: Given a dictionary of top target entropy values, and 2 indices (integers) of hyper and hyponyms, returns the SQLS row metric.
    Input:
    target_edict - a dictionary mapping target word indices to the entropy of their context distributions
    hyper/hypo - integer values corresponding to the indices of the words we're working with.
    
    Output:
    sqls_row - an integer.
    '''
    
    return 1 - target_edict[hyper]/target_edict[hypo]

def dif_sum(v1, v2):
    '''
    Generality metric: given two word vectors, returns the difference of their summations.
    
    Input:
    v1, v2: numpy arrays representing word vectors (can be DIVe or context vectors)
    
    Output: a float (unbounded)
    '''
    return v1.sum()-v2.sum()

def dif_norm(v1, v2):
    '''
    Generality metric: given two word vectors, returns the difference of their 2nd norms.
    
    Input:
    v1, v2: numpy arrays representing word vectors (DIVE or context)
    
    Output: a float (unbounded)
    '''
    return np.linalg.norm(v1, 2)-np.linalg.norm(v2,2)


def asymmetric_l1(hyperv, hypov, w0=5):
    '''
    Inclusion metric: returns the asymmetric l1 distance between two word vectors. This is kind of tricky 
    (proof to derive the estimator is in the appendix)
    
    Input:
    hyperv/hypov - word vectors for the hypernym and hyponym. Shape of (1,length(context)).
    w0 - hyperparameter specified in the paper. 
    
    Output:
    asymmetric_l1 - a float [0,1].
    '''
    #first, normalize both to get a proba distribution over contexts. Reshape too (to get)
    hyperv = hyperv.reshape(hyperv.shape[1])
    hypov = hypov.reshape(hypov.shape[1])
    hyperv_norm = hyperv/hyperv.sum()
    hypov_norm = hypov/hypov.sum()
    hypov_norm[hypov_norm==0] = 10**(-100) #replace 0 values with very small values, as specified in paper.

    #sort the elements by the values of v1_norm/v2_norm
    sorter = np.divide(-1*hyperv_norm, hypov_norm) #we add a negative to sort in reverse
    arr1inds = sorter.argsort()
    sorter_sorted = sorter[arr1inds[::-1]]*-1
    hypov_norm_sorted = hypov_norm[arr1inds[::-1]]
    
    s=0 ; c=0 
    u=np.zeros(len(hyperv))
    while((s <= 1) and (c<len(hyperv))):
        u_c = min(1-s, (w0+1)*hypov_norm_sorted[c])
        s += u_c
        u[c]=u_c
        c += 1
        
    asymmetric_l1 = 1 - np.sum(np.multiply(u,sorter_sorted))
    
    return asymmetric_l1

def read_evalution():
    '''
    Reads the evalution dataset, and returns the result as pandas dataframe with shape (7491,5) and with columns:
    ['relator', 'relation', 'relatum', 'pos_2', 'pos_1']
   
    '''
    target_url_relations = 'https://raw.githubusercontent.com/esantus/EVALution/master/EVALution_1.0/RELATIONS.txt'
    target_url_relata = 'https://raw.githubusercontent.com/esantus/EVALution/master/EVALution_1.0/RELATA.txt'
    dfs = []
    
    for target_url in [target_url_relations, target_url_relata]:
        fp = urllib.request.urlopen(target_url) # it's a file like object and works just like a file
        mybytes = fp.read()
        data = mybytes.decode('utf8')
        fp.close()
        data_l = data.split('\n')
        data_l_t = [k.split('\t') for k in data_l]
        dfs.append(pd.DataFrame(data_l_t))
        
    data_pd_relations, data_pd_relata = tuple(dfs)
    cols_relations = ['word', 'relation', 'relatum', 'tags', 'sent', 'vote_1', 'vote_2', 'vote_3', 'vote_4', 'vote_5', 'agreement',
           'subj_num', 'avg_score', 'var', 'score_min_v', 'source', 'source_score']
    data_pd_relations.columns = cols_relations
    
    cols_relata = ['relatum', 'tags', 'freq', 'dom_pos', 'pos_dist', 'form_dist', 'pos_dist_detailed']
    data_pd_relata.columns = cols_relata
    data_pd_relata_pos = data_pd_relata['dom_pos'].str.split('-', expand=True)
    data_pd_relata['pos'] = data_pd_relata_pos[1]
    
    relations_sub = data_pd_relations.loc[:, ['word', 'relation', 'relatum']]
    relata_sub = data_pd_relata.loc[:, ['relatum', 'pos']]
    _relations_relata = relations_sub.merge(relata_sub, on='relatum')
    relations_relata = _relations_relata.merge(relata_sub, left_on='word', right_on='relatum')
    relations_relata.drop('relatum_y', inplace=True, axis=1)
    relations_relata.columns = ['relator', 'relation', 'relatum', 'pos_2', 'pos_1']
    return relations_relata

def read_evalution2():
    '''
    Reads the second evalution dataset with the right number of samples
    '''
    target_url = 'https://raw.githubusercontent.com/bacumin/UnsupervisedHypernymy/master/datasets/EVALution.test'    
    fp = urllib.request.urlopen(target_url) # it's a file like object and works just like a file
    mybytes = fp.read()
    data = mybytes.decode('utf8')
    fp.close()
    data_l = data.split('\n')
    data_l_t = [k.split('\t') for k in data_l]
    df_pd = pd.DataFrame(data_l_t)
    
    df_pd.columns = ['word1', 'word2', 'hyp', 'relation']
    return df_pd
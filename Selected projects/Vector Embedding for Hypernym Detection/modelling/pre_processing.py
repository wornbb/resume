
import csv
import numpy as np
import scipy
from nltk.tokenize import sent_tokenize
import time
import math

def makeDictionary(wordTokens, cutoff=0):
    '''
    Given a source file it generates a dictionary with a unique token string as a key and the identification integer as a value

    Input:
    :filePath: path to the source file
    :cutoff: optional cutoff value to eliminate words with low frequency, if not initialized ano word is cutoff

    Output:
    :tokenMapping: a mapping from unique tokens/POS-tag pairing to unique integer identifications
    '''

    print("Getting Word Dictionary...", end='')

    tokenMapping = {}

    # count each token
    for t in wordTokens:
        tokenMapping[t] = tokenMapping.get(t, 0) + 1

    # remove words below cutoff and set value as index
    idx = 0
    for k, v in tokenMapping.items():
        if v < cutoff:
            del tokenMapping[k]
        else:
            tokenMapping[k] = idx
            idx += 1
    tokenMapping['UNK'] = len(tokenMapping)

    print(" Done")
    return tokenMapping


def getTokens(fileName):
    '''
    Gets list of tokens from file seperated by spaces

    Input:
    :fileName: path to file

    Output:
    :tokenMapping: list of word tokens
    '''

    print("Getting Tokens...", end='')

    with open(fileName) as f:
        textBody = f.read()

    print(" Done")
    return textBody.split()


def getCtxMatrix(tokens, dictionary, window):
    '''
    Given an ordered list of tokens, creates a sparse context matrix, returned as a lil_matrix.

    Input:
    :tokens: list of ordered word tokens
    :dictionary: word to numbered id mapping
    :window: the window size to consider when creating context matrix

    Output:
    :mat: a sparse context matrix, returned as a lil_matrix
    '''

    dim = len(dictionary)

    print("Creating Container Matrix of size %d..." % dim, end='')

    mat = scipy.sparse.lil_matrix((dim, dim), dtype=np.int)
    for i in range(dim):
        mat[i] = np.zeros(dim, dtype=np.int)

    print(" Done")

    print("Creating coo_matrix container")
    coo_mat = scipy.sparse.coo_matrix((dim, dim), dtype = np.int)
    print("Done")
    start = time.time()

    print("Building Context Matrix...", end='')

    for i in range(window, len(tokens) - window):

        lowerBound = i - window
        upperBound = i + window

        context = tokens[lowerBound:i] + tokens[i + 1:upperBound + 1]
        target = tokens[i]

        for c in context:
            if target in dictionary and c in dictionary:
                mat[dictionary[target], dictionary[c]] += 1

    print(' Finished in: %.2f sec' % (time.time() - start))

    return mat


def getPmiMatrix2(ctxt_matrix):
    '''
    Given a sparse context matrix, returns a sparse matrix with PMI score EFFICIENTLY. This is more efficient than pmi_matrix since
    it only iterates over non zero elements, and does this element-wise. No cutoff_0: this is a ppmi by default

    Input:
    :ctxt_matrix: sparse csr_matrix with co-occurrence counts.

    Output:
    :pmi_matrix: sparse pmi matrix with negative entries set to 0
    '''
    th = 0
    word_counts = ctxt_matrix.sum(axis=1).astype(np.float64)
    total_words = word_counts.sum()
    word_probas_arr = word_counts / total_words
    word_probas = {k: v for k, v in zip(range(ctxt_matrix.shape[0]), word_probas_arr)}
    cx = ctxt_matrix.tocoo().astype(np.float64)
    for i, j, v, k in zip(cx.row, cx.col, cx.data, range(len(cx.data))):
        p1 = word_probas[i]
        p2 = word_probas[j]
        if((p1 != 0) and (p2 != 0)):
            result = math.log((v / total_words) / (word_probas[i] * word_probas[j]))
    
            if (result > 0):
                cx.data[k] = result
            else:
                cx.data[k] = 0
        else:
            cx.data[k] = 0
    
    return scipy.sparse.csr_matrix(cx)


def getPmiMatrix(ctxMatrix, cutoff_0=True):
    '''
    Given a sparse context matrix, returns a sparse matrix with the Pointwise Mutual Information Score

    Input:
    :ctxMatrix: context Matrix
    :cutoff: optional cutoff flag to eliminate coocurrences that have a score below 0

    Output:
    :pmiMatrix: PMI Matrix

    '''

    print('Preparing to build PMI Matrix...', end='')

    # step 1: get the probability of each term in the matrix - sum over columns, then sum over sums
    wordCounts = ctxMatrix.sum(axis=1)
    totalWords = wordCounts.sum()
    wordProbas = wordCounts / totalWords
    pmiMatrix = scipy.sparse.csr_matrix(ctxMatrix.shape, dtype=np.float64)

    print(' Done')

    print('Building PMI Matrix...', end='')

    start = time.time()
    for i in range(ctxMatrix.shape[1]):  # row-wise operation
        rowPmi = np.log(np.divide(((ctxMatrix[i].toarray().T) / totalWords), wordProbas * wordProbas[i])).T

        if (cutoff_0):
            rowPmi[rowPmi < 0] = 0  # 0 cutoff

        pmiMatrix[i, :] = rowPmi
        pmiMatrix[i, :].eliminate_zeros()
        del rowPmi

    print(' Finished in: %.2f sec' % (time.time() - start))

    return pmiMatrix


def batchCreator(filePath, window, batchSize, dictionary, pmiMatrix):
    '''
    Given an input string (with spaced full stops " . " to delimit sentences), creates a sparse context matrix, returned as a lil_matrix.

    Input:
    :string: input string with spaced full stop to delimit sentences
    :num_words: integer indicating the top words to keep. Pass 0 if want to pass predefined index
    :window: the window size to consider when creating context matrix
    :word_list: if num_words is 0, pass a list of words to generate a context matrix on + a 10 cutoff of rare words.

    Output:
    :matrix: a sparse context matrix, returned as a lil_matrix
    :index: a python dictionary mapping word strings to integers
    '''

    print('Creating Batches...', end='')
    start = time.time()

    batches = []
    with open(filePath) as f:
        currentBatch = np.zeros((2, batchSize), dtype=np.int)
        trail = ""
        batchIndex = 0
        for chunk in f:
            batches, currentBatch, trail, batchIndex = createBatches(trail + chunk, window, batchSize, batchIndex, batches, currentBatch, dictionary, pmiMatrix)
            if (batchIndex > 0):
                lastBatch = np.zeros((2, batchIndex), dtype=np.int)
                lastBatch[0] = currentBatch[0][:batchIndex]
                lastBatch[1] = currentBatch[1][:batchIndex]
                batches.append(lastBatch)

    print("\nFinished in: %.2f sec" % (time.time() - start))
    return batches


def createBatches(chunk, window, batchSize, batchIndex, batches, batch, dictionary, pmiMatrix, pmiFilter=True):
    tokenWords = chunk.split()
    tokens = list(map(lambda t: dictionary[t] if t in dictionary else dictionary['UNK'], tokenWords))
    th = math.log(30)    

    for i in range(window, len(tokens) - window):

        lowerBound = i - window
        upperBound = i + window

        context = tokens[lowerBound:i] + tokens[i + 1:upperBound + 1]
        target = tokens[i]

        for c in context:

            if (not pmiFilter) or (pmiMatrix[target, c] > th):
                batch[0][batchIndex] = target
                batch[1][batchIndex] = c
                batchIndex += 1

                if batchIndex >= batchSize:
                    print(".", end='')
                    batches.append(batch)
                    batchIndex = 0
                    batch = np.zeros((2, batchSize), dtype=np.int)

    trail = " ".join(tokenWords[len(tokens) - window:])
    return (batches, batch, trail, batchIndex)


def main(fileName, window, batchSize, pmiFilter):
    tokens = getTokens(fileName)
    dictionary = makeDictionary(tokens)
    ctxMatrix = getCtxMatrix(tokens, dictionary, window)
    pmiMatrix = getPmiMatrix(ctxMatrix)
    batches = batchCreator(fileName, window, 100, dictionary, pmiMatrix, pmiFilter=pmiFilter)
    return batches


def script_main(fileName, window, batchSize, pmiFilter):
    tokens = getTokens(fileName)
    dictionary = makeDictionary(tokens)
    ctxMatrix = getCtxMatrix(tokens, dictionary, window)
    pmiMatrix = getPmiMatrix(ctxMatrix)
    batches = batchCreator(fileName, window, batchSize, dictionary, pmiMatrix, pmiFilter=pmiFilter)
    return batches, dictionary, pmiMatrix, ctxMatrix

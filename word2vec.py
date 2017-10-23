import gensim
import numpy as np
import scipy.sparse

class MeanEmbeddingVectorizer(object):
    def __init__(self, tokenizer, vocabulary):
        self.w2v = None
        self.dim = None
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        
    def fit(self, X):
        X = list(map(self.tokenizer, X))
        model = gensim.models.Word2Vec(X, size=100)
        self.w2v = dict(zip(model.wv.index2word, model.wv.syn0))
        self.dim = len(list(self.w2v.values())[0])
        return self
    
    def transform(self, X):
        res = np.array([
            np.mean([self.w2v[w] for w in words if w in self.w2v]
                or [np.zeros(self.dim)], axis=0)
            for words in X
            ])
        res = scipy.sparse.csr_matrix(res)
        return res

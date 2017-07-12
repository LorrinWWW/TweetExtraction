
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import *
from sklearn.feature_extraction.text import _document_frequency
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import normalize
from collections import namedtuple
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class DfidfTransformer(TfidfTransformer):
    
    def fit(self, X, y=None):
        if not sp.issparse(X):
            X = sp.csc_matrix(X)
        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)
            
            # perform idf smoothing if required
            #df += int(self.smooth_idf)
            #n_samples += int(self.smooth_idf)

            # log+1 instead of log makes sure terms with zero idf don't get
            # suppressed entirely.
            idf = np.log(1 + 1/df)
            
            self._df_diag = sp.spdiags(df, diags=0, m=n_features,
                                       n=n_features, format='csr')
            self._idf_diag = sp.spdiags(idf, diags=0, m=n_features,
                                        n=n_features, format='csr')
        return self
    
    def transform(self, X, copy=True):
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            # preserve float family dtype
            X = sp.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)

        n_samples, n_features = X.shape

        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1

        if self.use_idf:
            check_is_fitted(self, '_idf_diag', 'idf vector is not fitted')
            check_is_fitted(self, '_df_diag', 'idf vector is not fitted')

            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            # *= doesn't work
            X = X * self._idf_diag * self._df_diag

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        return X

class TfIdf(object):
    
    def __init__(self, corpus):
        self.corpus = corpus
        self.vecs = []
        self.words = []
        self._vectorizer = None
        self.train()
        
    def train(self):
        # tf-idf, word[j],tfidf[i, j]
        #该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频 
        vectorizer=CountVectorizer() 
        #该类会统计每个词语的tf-idf权值 
        transformer=TfidfTransformer() 
        #第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵  
        vecs=transformer.fit_transform(vectorizer.fit_transform(self.corpus))
        #获取词袋模型中的所有词语
        words=vectorizer.get_feature_names()
        
        self._transformer = transformer
        self._vectorizer = vectorizer
        self.vecs = vecs
        self.words = words
        
    def vectorizer(self, post):
        return self._vectorizer.transform([post])
    
class DfIdf(object):
    
    def __init__(self, corpus):
        self.corpus = corpus
        self.vecs = []
        self.words = []
        self._vectorizer = None
        self.train()
        
    def train(self):
        # tf-idf, word[j],tfidf[i, j]
        #该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频 
        vectorizer=CountVectorizer() 
        #该类会统计每个词语的tf-idf权值 
        transformer=DfidfTransformer() 
        #第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵  
        vecs=transformer.fit_transform(vectorizer.fit_transform(self.corpus))
        #获取词袋模型中的所有词语
        words=vectorizer.get_feature_names()
        
        self._transformer = transformer
        self._vectorizer = vectorizer
        self.vecs = vecs
        self.words = words
        
    def vectorizer(self, post):
        return self._vectorizer.transform([post])
        
        
class Word2Vec(object):
    
    def __init__(self, corpusl, size):
        self.corpusl = corpusl
        self.model = None
        self.vecs = []
        self.words = []
        self.size = size
        self.train()
        
    def train(self):
        model = gensim.models.Word2Vec(self.corpusl, size=self.size, workers=4)
        self.model = model
        self.vecs = []
        for c in self.corpusl:
            vec = np.zeros(self.size)
            for w in c:
                if w in model:
                    tmp = model[w] / np.linalg.norm(model[w])
                    vec += tmp
            self.vecs.append(vec)
    
    def vectorizer(self, post):
        vec = np.zeros(self.size)
        for w in post:
            if w in self.model:
                tmp = self.model[w] / np.linalg.norm(self.model[w])
                vec += tmp
        return vec

class Doc2Vec(object):
    
    def __init__(self, corpusl, size):
        self.corpusl = corpusl
        self.model = None
        self.vecs = []
        self.size = size
        self.train()
        
    def train(self):
        docs = []
        analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
        for i, text in enumerate(self.corpusl):
            tags = [i]
            docs.append(analyzedDocument(text, tags))
            
        model = gensim.models.Doc2Vec(docs, size=self.size, workers=4)
        self.model = model
        self.vecs = model.docvecs
        
    def vectorizer(self, post):
        return self.model.infer_vector(post)
                    
                    
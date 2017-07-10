
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer 
from collections import namedtuple
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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
                    
                    
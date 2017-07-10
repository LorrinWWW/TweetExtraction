import numpy as np
import scipy.sparse as sparse
from datetime import datetime


def normFor1DSparseMatrix(m):
    if sparse.issparse(m):
        return np.sqrt(sum(v*v for v in m.data))
    return np.linalg.norm(m)

def normFor1DSparseMatrixNonZero(m):
    norm = normFor1DSparseMatrix(m)
    if norm == 0:
        return -1
    return norm

class Tweet(object):
    
    def __init__(self, post_time, content, poster_id, poster_url, repost_num, comment_num):
        self.post_time = datetime.strptime(post_time, '%Y/%m/%d %H:%M').timestamp()
        self.content = content
        self.poster_id = poster_id
        self.poster_url = poster_url
        self.repost_num = repost_num
        self.comment_num = comment_num

class TweetVector(object):
    
    def __init__(self, tvi, tsi, wi, t):
        self.tvi = tvi
        self.tsi = tsi
        self.wi = wi
        
        self.t = t
    

#from time import clock 
#t_sum_v = 0
#t_wsum_v = 0
#t_ts1 = 0
#t_ts2 = 0
class TSV(object):
    
    def __init__(self, cluster, m=50):
        self.m = m
        if len(cluster) > 1200:
            cluster = np.random.choice(cluster, 1200, replace=False)
        self.sum_v = sum(t.tvi / normFor1DSparseMatrixNonZero(t.tvi) for t in cluster)
        self.wsum_v = sum(t.tvi * t.wi for t in cluster)
        self.ts1 = sum(t.tsi for t in cluster)
        self.ts2 = sum(t.tsi * t.tsi for t in cluster)
        self.n = len(cluster)
        
        # self.cluster = cluster
        # self.cv = self.wsum_v / self.n
        self.cv = self.sum_v / self.n # !! 论文中是wsum
        self.ft_set = cluster
        self.updateFtSet()
    
    def updateFtSet(self):
        if self.m < len(self.ft_set):
            self.ft_set = sorted(self.ft_set, key=lambda tv: normFor1DSparseMatrix(tv.tvi-self.cv))[:self.m]
        
    def d2centre(self, tv):
        return normFor1DSparseMatrix(tv.tvi - self.cv)
    
    def maxSim(self, tv):
        return self.cosine_similarity(tv.tvi, self.cv)
    
    def mbs(self, beta):
        if sparse.issparse(self.sum_v):
            return beta * (sparse.csr_matrix.dot(self.sum_v, self.wsum_v.transpose()) / 
                           (self.n * normFor1DSparseMatrixNonZero(self.wsum_v)))[0,0]
        return beta * (np.dot(self.sum_v, self.wsum_v)) / (self.n * normFor1DSparseMatrixNonZero(self.wsum_v))
    
    def cosine_similarity(self, v1, v2):
        if sparse.issparse(v1):
            return (sparse.csr_matrix.dot(v1, v2.transpose()) / 
                    (normFor1DSparseMatrixNonZero(v1) * normFor1DSparseMatrixNonZero(v2)))[0,0]
        return (np.dot(v1, v2)) / (normFor1DSparseMatrixNonZero(v1) * normFor1DSparseMatrixNonZero(v2))
    
    def addTv(self, tv):
        self.sum_v += tv.tvi
        self.wsum_v += tv.tvi * tv.wi
        self.ts1 += tv.tsi
        self.ts2 += tv.tsi * tv.tsi
        self.n += 1
        self.cv = self.sum_v / self.n # !! 论文中是wsum
        
        self.ft_set.append(tv)
        updateFtSet()
        
    def aggregationFrom(self, tsv):
        self.sum_v += tsv.sum_V
        self.wsum_v += tsv.wsum_v
        self.ts1 += tsv.ts1
        self.ts2 += tsv.ts2
        self.n += tsv.n
        self.cv = self.sum_v / self.n # !! 论文中是wsum
        
        self.ft_set += tsv.ft_set
        updateFtSet()
        
    def subtractionFrom(self, tsv):
        self.sum_v -= tsv.sum_V
        self.wsum_v -= tsv.wsum_v
        self.ts1 -= tsv.ts1
        self.ts2 -= tsv.ts2
        self.n -= tsv.n
        self.cv = self.sum_v / self.n # !! 论文中是wsum
        # updateFtSet() # sub没办法更新ftset
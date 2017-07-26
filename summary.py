import jieba
import numpy as np
import scipy.sparse as sparse
from math import sqrt
from snownlp import normal
from snownlp.summary import textrank
from sumy.summarizers.lex_rank import LexRankSummarizer
from datatype import *

class LRS(LexRankSummarizer):
    def __call__(self, tsv, sentences_count):
        docs = [tv.t.content for tv in tsv.ft_set]
        tvs = [tv.tvi for tv in tsv.ft_set]

        matrix = self._create_matrix(tvs, self.threshold)
        scores = self.power_method(matrix, self.epsilon)
        ratings = dict(zip(docs, scores))

        return self._get_best_sentences(docs, sentences_count, ratings)
    
    def _create_matrix(self, tvs, threshold):
        tvs_count = len(tvs)
        matrix = np.zeros((tvs_count, tvs_count))
        degrees = np.zeros((tvs_count, ))

        for row, tv1 in enumerate(tvs):
            for col, tv2 in enumerate(tvs):
                matrix[row, col] = self.cosine_similarity(tv1, tv2)

                if matrix[row, col] > threshold:
                    matrix[row, col] = 1.0
                    degrees[row] += 1
                else:
                    matrix[row, col] = 0

        for row in range(tvs_count):
            for col in range(tvs_count):
                if degrees[row] == 0:
                    degrees[row] = 1

                matrix[row][col] = matrix[row][col] / degrees[row]

        return matrix

    def cosine_similarity(self, v1, v2):
        if sparse.issparse(v1):
            return (v1 * v2.transpose() / 
                    (normFor1DSparseMatrixNonZero(v1) * normFor1DSparseMatrixNonZero(v2)))[0,0]
        return (np.dot(v1, v2)) / (normFor1DSparseMatrixNonZero(v1) * normFor1DSparseMatrixNonZero(v2))
    
class TSVRS(LRS):
    def __call__(self, tsv, sentences_count, lbd=0.5, nmax=50):
        self._ensure_dependencies_installed()
        
        docs = [tv.t.content for tv in tsv.ft_set]
        tvs = [tv.tvi for tv in tsv.ft_set]

        matrix = self._create_matrix(tvs, self.threshold)
        scores = self.power_method(matrix, self.epsilon)
        ratings = dict(zip(tsv.ft_set, scores))

        best_tvs = []
        best_tvs.append(self._get_best_sentences(tsv.ft_set, 1, ratings)[0])
        
        for _ in range(sentences_count - 1):
            vmax = -9999
            tvmax = None
            for tv, score in ratings.items():
                if tv in best_tvs:
                    continue
                v = len(tsv.ft_set)/nmax*score - lbd*np.mean([self.cosine_similarity(tv.tvi, tv2.tvi) for tv2 in best_tvs])
                if v > vmax:
                    vmax = v
                    tvmax = tv
            if tvmax is not None:
                best_tvs.append(tvmax)
        return [tv.t.content for tv in best_tvs]
        
def summaryTSVRank(tsv, limit=5, lbd=0.5, nmax=50, threshold=0.1):
    '''
    docs = [
        '第一篇微博',
        '第二篇微博',
        ...
    ]
    '''
    summarizer = TSVRS()
    summarizer.threshold = threshold
    summary = summarizer(tsv, limit, lbd=lbd, nmax=nmax) #Summarize the document with 5 sentences
    return summary

def summaryLexRank(tsv, limit=5, nmax=50, threshold=0.1):
    '''
    docs = [
        '第一篇微博',
        '第二篇微博',
        ...
    ]
    '''
    summarizer = LRS()
    summarizer.threshold = threshold
    summary = summarizer(tsv, limit) #Summarize the document with 5 sentences
    return summary

def summaryOfSents(docs, limit=5):
    '''
    docs = [
        '第一篇微博',
        '第二篇微博',
        ...
    ]
    '''
    merged_doc = []
    sents = []
    for doc in docs:
        sents_t = normal.get_sentences(doc)
        sents += sents_t
        for sent in sents_t:
            words = jieba.lcut(sent)
            words = normal.filter_stop(words)
            merged_doc.append(words)
    
    rank = textrank.TextRank(merged_doc)
    rank.solve()
    ret = []
    for index in rank.top_index(limit):
        ret.append(sents[index])
    return ret

def summaryTextRank(docs, limit=5):
    '''
    docs = [
        '第一篇微博',
        '第二篇微博',
        ...
    ]
    '''
    merged_doc = []
    for doc in docs:
        words = jieba.lcut(doc)
        words = normal.filter_stop(words)
        merged_doc.append(words)
    
    rank = textrank.TextRank(merged_doc)
    rank.solve()
    ret = []
    for index in rank.top_index(limit):
        ret.append(docs[index])
    return ret

if __name__ == '__main__':
    # summaryOfSents(['你好NLP。今天买了手机', '测试句子。今天我买了手机。','今天我买了小米手机。','买手机', '手机'])
    summaryOfDocs(['你好NLP。今天买了手机', '测试句子。今天我买了手机。','今天我买了小米手机。','买手机', '手机'])
    
    
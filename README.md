# TweetExtraction

This is the partial implementation of [On summarization and timeline generation for evolutionary tweet stream](http://ieeexplore.ieee.org/iel7/69/4358933/06871372.pdf).

And this is done in limited time thus the performance is not well optimised.

Sorry that most parts of commons are wirtten in Chinese, do not hesitate to ask me. And maybe I will rewrite in Enlish and polish the codes in the future.

## Getting started
1. Find a dataset of tweets or weibos, save it to folder ./data. You can modify filereader.py to adapt your case;
2. Run each block in run.ipynb;

## Overview
### 静态部分
1. 文本预处理(分词、停用词等)
2. Tf-Idf将文本向量化
3. kmeans初步聚类，每一个cluster用一个TSV来表示
4. 对每个TSV摘要
    - LexRank(或者其他)获得与该TSV中心最接近的Tweet
    - 为防止获得的摘要内容过于接近，采用以下办法
    - 按公式，获得尽可能接近TSV中心、又与之前得到的推特相似度尽可能小的Tweet，其中$\lambda$是可以调节的参数
    $$
      t = argmax_{t_i} [ \lambda \frac{n_{t_i}}{n_{max}} LR(t_i) - (1-\lambda)avg_{t_j \in S}Sim(t_i, t_j)  ]
    $$
    - 往复上述步骤，获得若干条高度概括，内容有所不同的Tweet
    
### 动态部分
1. Pyramidal Time Frame
2. 新的Tweet进来，寻找相似度最大cluster，比较“minimum bounding similarity”，
    - 若MBS<Sim(new tweet)，则将该tweet加入该cluster；
    - 若MBS>Sim(new tweet)，则新建一个cluster；
3. 我们认为话题的时间分布是符合正态的，已知tsv可计算时间均值和方差，故可以此估计话题时间线是否已到达正态曲线的末端
4. 当cluster的数量过多时，我们需要进行合并

5. **timeline : variations方差**
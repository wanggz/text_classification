'''
@Author: xiaoyao jiang
@Date: 2020-04-08 17:22:54
LastEditTime: 2021-01-03 21:39:08
LastEditors: Peixin Lin
@Description: train embedding & tfidf & autoencoder
FilePath: /JD_NLP1-text_classfication/embedding.py
'''
import pandas as pd
import numpy as np
from gensim import models
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import joblib
import jieba
from gensim.models import LdaMulticore
from features import label2idx
import gensim
import config


class SingletonMetaclass(type):
    '''
    @description: singleton
    '''
    def __init__(self, *args, **kwargs):
        self.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.__instance is None:
            self.__instance = super(SingletonMetaclass,
                                    self).__call__(*args, **kwargs)
            return self.__instance
        else:
            return self.__instance


class Embedding(metaclass=SingletonMetaclass):
    def __init__(self):
        '''
        @description: This is embedding class. Maybe call so many times. we need use singleton model.
        In this class, we can use tfidf, word2vec, fasttext, autoencoder word embedding
        @param {type} None
        @return: None
        '''

        #######################################################################
        #  1        TODO:  读取停用词 #
        #######################################################################
        # 读取停止词
        #
    def read_stopwords_file(self):
        stopWords = [x.strip() for x in open('./data/stopwords.txt').readlines()]
        return stopWords

    def load_data(self, path):
        '''
        @description:Load all data, then do word segment
        @param {type} None
        @return:None
        '''
        data = pd.read_csv(path, sep='\t')
        data = data.fillna("")
        #######################################################################
        #  2        TODO:  去除停用词 #
        #######################################################################
        # 对data['text']中的词进行分割，并去除停用词 参考格式： data['text'] = data["text"].apply(lambda x: " ".join(x))
        #
        stopwords = self.read_stopwords_file()
        data['text'] = data["text"].apply(lambda x: " ".join([w for w in x.split() if w not in stopwords and w != '']))

        self.labelToIndex = label2idx(data)
        data['label'] = data['label'].map(self.labelToIndex)
        #data['label'] = data.apply(lambda row: float(row['label']), axis=1)
        data = data[['text', 'label']]
        
#         self.train, _, _ = np.split(data[['text', 'label']].sample(frac=1), [int(data.shape[0] * 0.7), int(data.shape[0] * 0.9)])
        self.train = data['text']



    def trainer(self):
        '''
        @description: Train tfidf,  word2vec, fasttext and autoencoder
        @param {type} None
        @return: None
        '''
        #######################################################################
        #  3        TODO:  模型训练 # tfidf
        #######################################################################
        #count_vect 对 tfidfVectorizer 初始化
        #
        # word count feature
        count_vect = CountVectorizer()
        train_counts = count_vect.fit_transform(self.train)
        # tf-idf feature
        tfidf_transformer = TfidfTransformer()
        self.tfidf = tfidf_transformer.fit_transform(train_counts)

        #######################################################################
        #  4        TODO:  模型训练 # w2v
        #######################################################################
        #对 w2v 初始化 并建立词表，训练
        #
        w2v_train = []
        for sentence in self.train.tolist():
            w2v_train.append(sentence.split())
        model_size = 300
        self.w2v = Word2Vec(w2v_train, size=model_size, min_count=1)

        self.id2word = gensim.corpora.Dictionary(w2v_train)
        corpus = [self.id2word.doc2bow(text) for text in w2v_train]

        #######################################################################
        #  5        TODO:  模型训练 #
        #######################################################################
        # 建立LDA模型
        #
        self.LDAmodel = gensim.models.ldamulticore.LdaMulticore(
            corpus=corpus,
            num_topics=11,
            id2word=self.id2word,
            chunksize=100,
            workers=7, # Num. Processing Cores - 1
            passes=50,
            eval_every = 1,
            per_word_topics=True)

    def saver(self):
        '''
        @description: save all model
        @param {type} None
        @return: None
        '''
        joblib.dump(self.tfidf, './model/tfidf')

        self.w2v.wv.save_word2vec_format('./model/w2v.bin',
                                         binary=False)

        self.LDAmodel.save('./model/lda')

    def load(self):
        '''
        @description: Load all embedding model
        @param {type} None
        @return: None
        '''
        self.tfidf = joblib.load('./model/tfidf')
        self.w2v = models.KeyedVectors.load_word2vec_format('./model/w2v.bin', binary=False)
        self.lda = models.ldamodel.LdaModel.load('./model/lda')


if __name__ == "__main__":
    em = Embedding()
    em.load_data(config.train_data_file)
    em.trainer()
    em.saver()

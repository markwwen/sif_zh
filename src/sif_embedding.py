# -*- coding: utf-8 -*-
import numpy as np
import jieba
import pickle
import gensim

w2v_model = gensim.models.Word2Vec.load('/data/w2v_model/tea/with_wiki.model')



def get_weighted_embedding(sentences, w2v_model, dict_word_weight, w2v_size):
    sent_embedding = []
    for s in sentences:
        s_embedding = np.array([0.0] * w2v_size)
        words = list(jieba.cut(s))
        for word in words:
            s_embedding += w2v_model[word] * dict_word_weight[word]
        sent_embedding.append(s_embedding)
    return sent_embedding


if __name__ == '__main__':
    sentences = ['这杯茶真好喝', '我今天不开心']
    dict_word_weight = pickle.load(open('../data/dict_word_weight.p', 'rb'))
    weighted_embeding = get_weighted_embedding(sentences, w2v_model, dict_word_weight, 200)
    print(weighted_embeding[0])
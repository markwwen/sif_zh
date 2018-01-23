# -*- coding: utf-8 -*-
import numpy as np
import jieba
import pickle
import gensim
from sklearn.decomposition import TruncatedSVD
from scipy import spatial

# set the word and sentence dimension
tea_w2v_size = 200

# get the word2vec model
tea_w2v_model = gensim.models.Word2Vec.load('/data/w2v_model/tea/with_wiki.model')

# get the p(w) dict
tea_dict_word_weight = pickle.load(open('/data/sif_model/dict_word_weight.p', 'rb'))

# get the sif_embedding of sentences
tea_sif_embedding = pickle.load(open('/data/sif_model/sif_embedding.p', 'rb'))

# get the principle component
tea_pc = pickle.load(open('/data/sif_model/pc.p', 'rb'))

# get the sentences
tea_sentences = []
with open('/data/sif_model/tea_question.csv') as f:
    for line in f.readlines():
        line = line.replace('\n', '')
        tea_sentences.append(line)


def get_weighted_embedding(sentences, w2v_model, dict_word_weight, w2v_size):
    sent_embedding = []
    for s in sentences:
        s_embedding = np.array([0.0] * w2v_size)
        words = list(jieba.cut(s))
        for word in words:
            if word in w2v_model:
                s_embedding += w2v_model[word] * dict_word_weight[word]
            else:
                s_embedding += [1] * w2v_size
        sent_embedding.append(s_embedding)
    return np.array(sent_embedding)


def compute_pc(X, npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_


def remove_pc(X, npc=1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    pc = compute_pc(X, npc)
    if npc == 1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX


# return the cosine similarity of v1 and v2
def my_cosine_similarity(v1, v2):
    return 1 - spatial.distance.cosine(v1, v2)


def get_sent_embedding(text, pc, w2v_model, dict_word_weight, w2v_size):
    # init the sentence embedding
    s_embedding = np.array([0.0] * w2v_size)
    words = list(jieba.cut(text))
    for word in words:
        if word in w2v_model:
            s_embedding += w2v_model[word] * dict_word_weight[word]
        else:
            s_embedding += [1] * w2v_size
    # remove the principle component from the sentence
    rmpc_s_embedding = s_embedding - s_embedding.dot(pc.transpose()) * pc
    return rmpc_s_embedding


def get_most_similar_k(test_str, k, sentences, sentences_emb, w2v_model, dict_word_weight, pc, w2v_size):
    sent_emb = get_sent_embedding(test_str, pc, w2v_model, dict_word_weight, w2v_size)
    distance_list = np.array(list(map(lambda x: my_cosine_similarity(sent_emb, x), sentences_emb)))
    smallest_k_index = distance_list.argsort()[::-1][:k]
    for i in smallest_k_index:
        print(sentences[i], distance_list[i])


if __name__ == '__main__':
    # print(len(sentences))
    # print(sentences[:2])
    # get the weighted embedding
    # weighted_embedding = get_weighted_embedding(sentences, w2v_model, dict_word_weight, 200)
    # pickle.dump(weighted_embedding, open('../data/sentences_weighted_embedding.p', 'wb'))
    # weighted_embedding = pickle.load(open('../data/sentences_weighted_embedding.p', 'rb'))

    # get the pc
    # pc = compute_pc(weighted_embedding)
    # pickle.dump(pc, open('../data/pc.p', 'wb'))
    # pc = pickle.load(open('../data/pc.p', 'rb'))
    # print(pc)

    # get the sif_embedding of sentences
    # sif_embedding = remove_pc(weighted_embedding)
    # pickle.dump(sif_embedding, open('../data/sif_embedding.p', 'wb'))

    test_sent = '是不是等级越高的茶越好'
    get_most_similar_k(test_sent, 5, tea_sentences, tea_sif_embedding, tea_w2v_model, tea_dict_word_weight, tea_pc, tea_w2v_size)

# -*- coding: utf-8 -*-
import numpy as np
import jieba
import pickle
import gensim
from . import params
from sklearn.decomposition import TruncatedSVD
from numpy import dot
from numpy.linalg import norm

# the path that you store you word2vec model and other sif model
base_data_path = '/data'

# set the word and sentence dimension
tea_w2v_size = 200
# get the word2vec model
tea_w2v_model = gensim.models.Word2Vec.load(base_data_path + '/w2v_model/tea/tea.model')
# get the p(w) dict
tea_dict_word_weight = pickle.load(open(base_data_path + '/sif_model/dict_word_weight.p', 'rb'))
# get the principle component
tea_pc = pickle.load(open(base_data_path + '/sif_model/pc.p', 'rb'))
# get the sif_embedding of sentences
tea_sif_embedding = pickle.load(open(base_data_path + '/sif_model/sif_embedding.p', 'rb'))
# get the sentences
tea_sentences = []


with open(base_data_path + '/sif_model/tea_question.csv') as f:
    for line in f.readlines():
        line = line.replace('\n', '')
        tea_sentences.append(line)

o_params = params.params()
o_params.w2v_size = tea_w2v_size
o_params.w2v_model = tea_w2v_model
o_params.dict_word_weight = tea_dict_word_weight
o_params.pc = tea_pc
o_params.sif_embedding = tea_sif_embedding
o_params.sentences = tea_sentences

def build_pc_and_sif_embedding(the_params):
    """
    build the weighted embedding, principle component and sif embedding and save in pickle.

    :param the_params: params class instance, which contain all the parameters
    :return: null
    """
    weighted_embedding = get_weighted_embedding(the_params)
    pickle.dump(weighted_embedding, open(base_data_path + '/sif_model/weighted_embedding.p', 'wb'))

    pc = compute_pc(weighted_embedding)
    pickle.dump(pc, open(base_data_path + '/sif_model/pc.p', 'wb'))

    sif_embedding = remove_pc(weighted_embedding)
    pickle.dump(sif_embedding, open(base_data_path + '/sif_model/sif_embedding.p', 'wb'))



def get_weighted_embedding(the_params):
    """

    :param the_params: params class instance, which contain all the parameters
    :return: the weighted embedding of sentences
    """
    weighted_embedding = []
    for s in the_params.sentences:
        s_embedding = np.array([0.0] * the_params.w2v_size)
        words = list(jieba.cut(s))
        for word in words:
            if word in the_params.w2v_model:
                s_embedding += the_params.w2v_model[word] * the_params.dict_word_weight[word]
            else:
                s_embedding += [1] * the_params.w2v_size
        weighted_embedding.append(s_embedding)
    return np.array(weighted_embedding)


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
    cos_sim = dot(v1, v2) / (norm(v1) * norm(v2))
    return cos_sim[0]


def get_sent_embedding(text, the_params):
    """

    :param the_params: params class instance, which contain all the parameters
    :return: the SIF embedding for the sentence
    """
    # init the sentence embedding
    s_embedding = np.array([0.0] * the_params.w2v_size)
    words = list(jieba.cut(text))
    for word in words:
        if word in the_params.w2v_model:
            s_embedding += the_params.w2v_model[word] * the_params.dict_word_weight[word]
        else:
            s_embedding += [1] * the_params.w2v_size
    # remove the principle component from the sentence
    rmpc_s_embedding = s_embedding - s_embedding.dot(the_params.pc.transpose()) * the_params.pc
    return rmpc_s_embedding


def get_most_similar_k(text, k, the_params):
    """

    :param the_params: params class instance, which contain all the parameters
    :return: the most similar k sentences and its cosine similarity
    """
    similarity_sentences = []
    sent_emb = get_sent_embedding(text, the_params)
    # distance_list = np.array(list(map(lambda x: my_cosine_similarity(sent_emb, x), sentences_emb)))
    distance_list = np.array([my_cosine_similarity(sent_emb, x) for x in the_params.sif_embedding])
    smallest_k_index = distance_list.argsort()[::-1][:k]
    for i in smallest_k_index:
        similarity_sentences.append([the_params.sentences[i], distance_list[i]])
    return similarity_sentences


if __name__ == '__main__':
    build_pc_and_sif_embedding(o_params)

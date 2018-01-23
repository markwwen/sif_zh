# -*- coding: utf-8 -*-
import pickle


base_data_path = '/data'


def get_dict_word_fre():
    word_all_num = 0
    dict_word_num = {}
    dict_word_fre = {}
    with open(base_data_path + '/word2vec_with_wiki.corpus') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            words = line.split(' ')
            for word in words:
                word_all_num += 1
                if word in dict_word_num:
                    dict_word_num[word] += 1
                else:
                    dict_word_num[word] = 1

    for word in dict_word_num:
        dict_word_fre[word] = dict_word_num[word] / word_all_num

    return word_all_num, dict_word_fre


def get_dict_word_weight(dict_word_fre, a=1e-3):
    if a <= 0:
        a = 1.0

    dict_word_weight = {}
    for word in dict_word_fre:
        dict_word_weight[word] = a / (a + dict_word_fre[word])
    return dict_word_weight



if __name__ == '__main__':
    # word_all_num, dict_word_fre= get_word_fre()
    # pickle.dump(dict_word_fre, open('../data/dict_word_fre.p', 'wb'))
    dict_word_fre = pickle.load(open(base_data_path + '/dict_word_fre.p', 'rb'))
    print(dict_word_fre['好'])
    print(dict_word_fre['的'])

    # dict_word_weight = get_word_weight(dict_word_fre)
    # pickle.dump(dict_word_weight, open('../data/dict_word_weight.p', 'wb'))
    dict_word_weight = pickle.load(open(base_data_path + '/dict_word_weight.p', 'rb'))
    print(dict_word_weight['好'])
    print(dict_word_weight['的'])


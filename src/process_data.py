# -*- coding: utf-8 -*-
import pickle



corpus = './word2vec_with_wiki.corpus'


def get_words_fre():
    word_all_num = 0
    dict_word_num = {}
    dict_word_fre = {}
    with open('../data/word2vec_with_wiki.corpus') as f:
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



if __name__ == '__main__':
    # word_all_num, dict_word_fre= get_words_fre()
    # pickle.dump(dict_word_fre, open('../data/dict_word_fre.p', 'wb'))
    dict_word_fre = pickle.load(open('../data/dict_word_fre.p', 'rb'))
    print(dict_word_fre['好'])
    print(dict_word_fre['的'])


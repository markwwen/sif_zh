# -*- coding: utf-8 -*-
from src.params import Params
from src.sif_embedding import build_pc_and_sif_embedding_list
from src.sif_embedding import get_most_similar_k

if __name__ == '__main__':
    p = Params()
    p.tea_init()

    # build the model only if in the first time you
    build_pc_and_sif_embedding_list(p)

    test_sent = '是不是等级越高的茶越好？'
    similarity_sentences = get_most_similar_k(test_sent, 5, p)
    for i in similarity_sentences:
        print(i)

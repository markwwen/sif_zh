from src.sif_embedding import *

if __name__ == '__main__':
    test_sent = '是不是等级越高的茶越好'
    similarity_sentences = get_most_similar_k(test_sent, 5, the_params=o_params)
    for i in similarity_sentences:
        print(i)

from src.sif_embedding import get_most_similar_k
from src.params import Params



if __name__ == '__main__':
    p = Params()
    p.tea_init()
    test_sent = '是不是等级越高的茶越好？'
    similarity_sentences = get_most_similar_k(test_sent, 5, p)
    for i in similarity_sentences:
        print(i)

from . import params
import pickle
import gensim



if __name__ == '__main__':

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

    pass
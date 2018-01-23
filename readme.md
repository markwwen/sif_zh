# SIF_YIMIAN

This is the implement of the sentence embedding algorithm in [the paper](https://openreview.net/forum?id=SyK00v5xx) "A Simple but Tough-to-Beat Baseline for Sentence Embeddings" in Python3 and in Chinese corpus.


## Install

```angular2html
$ pip install -r requirements.txt
```

## Get started
To get started, you need:
- A corpus to train word2vec model and get frequency of word
- A corpus of sentences (here is some question about tea in Chinese)

You can use the function `get_dict_word_fre` in `process_data.py` to get a `dict` of word frequency,
and the function in `sif_embedding.py` will help you to calculate the weighted-embedding of sentences, principle
component and final sif-embedding, and save in pickle format.

If you need to run the code, please make sure the model is in the right path.




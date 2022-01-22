import pandas as pd
import numpy as np
from matplotlib import pyplot
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle

df_train = pd.read_csv(
    r"D:\Studie\Business Analytics\Applied Text Mining\assignment3\train\pre_processed_train.csv",
)

num_sentences = list(set(df_train["idx_sent"]))
word_set = set(df_train["Token"])


sentences = []
for sent_id in num_sentences:
    sentences.append(list(df_train[df_train.idx_sent == sent_id]["Token"]))


model_2 = Word2Vec(vector_size=300, min_count=1, sg=1)
model_2.build_vocab(sentences)
training_examples_count = model_2.corpus_count
model_2.train(sentences, total_examples=training_examples_count, epochs=model_2.epochs)
model_2.wv.save_word2vec_format("word2vec_model_vectors2.bin")


# Use pre trained embeddings
google_wv = KeyedVectors.load_word2vec_format(
    r"D:\Studie\Business Analytics\Applied Text Mining\assignment3\GoogleNews-vectors-negative300.bin",
    binary=True,
)

word_vectors = {}

for sentence in sentences:
    for word in sentence:
        try:
            word_vectors[word] = google_wv[word]
        except:
            word_vectors[word] = model_2.wv[word]


with open("VectorsFromPretraind.pickle", "wb") as handle:
    pickle.dump(word_vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)

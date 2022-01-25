import pandas as pd
import numpy as np
from matplotlib import pyplot
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle

df_train = pd.read_csv(
    r"D:\Studie\Business Analytics\Applied Text Mining\assignment3\Data\tmtrain.Preprocessed.csv",
)
df_dev = pd.read_csv(
    r"D:\Studie\Business Analytics\Applied Text Mining\assignment3\Data\tmdev.Preprocessed.csv",
)
df_test_circle = pd.read_csv(
    r"D:\Studie\Business Analytics\Applied Text Mining\assignment3\Data\test\SEM-2012-SharedTask-CD-SCO-test-circle.Preprocessed.csv",
)
df_test_cardboard = pd.read_csv(
    r"D:\Studie\Business Analytics\Applied Text Mining\assignment3\Data\test\SEM-2012-SharedTask-CD-SCO-test-cardboard.Preprocessed.csv",
)

list_of_dfs = [df_train, df_dev, df_test_circle, df_test_cardboard]


combined_tokens = pd.concat(
    [df_train.Token, df_dev.Token, df_test_circle.Token, df_test_cardboard.Token]
)


word_set = set(combined_tokens)


sentences = []
for df in list_of_dfs:
    num_sentences = list(set(df["idx_sent"]))
    for sent_id in num_sentences:
        sentences.append(list(df[df.idx_sent == sent_id]["Token"]))


model_2 = Word2Vec(vector_size=300, min_count=1, sg=1)
model_2.build_vocab(sentences)
training_examples_count = model_2.corpus_count
model_2.train(sentences, total_examples=training_examples_count, epochs=model_2.epochs)
model_2.wv.save_word2vec_format("word2vec_model_vectors2.bin")

model_2.wv["looking-glass"]

# Use pre trained embeddings
google_wv = KeyedVectors.load_word2vec_format(
    r"D:\Studie\Business Analytics\Applied Text Mining\assignment3\GoogleNews-vectors-negative300.bin",
    binary=True,
)

word_vectors = {}

for word in word_set:
    try:
        word_vectors[word] = google_wv[word]
    except:
        try:
            word_vectors[word] = model_2.wv[word]
        except:
            word_vectors[word] = [0] * 300


with open("VectorsFromPretraind.pickle", "wb") as handle:
    pickle.dump(word_vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)

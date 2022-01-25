import pandas as pd
import numpy as np
from matplotlib import pyplot
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle

dict_vectors = pd.read_pickle(
    r"D:\Studie\Business Analytics\Applied Text Mining\assignment3\code\VectorsFromPretraind.pickle"
)

df_dev = pd.read_csv(
    r"D:\Studie\Business Analytics\Applied Text Mining\assignment3\Data\test\SEM-2012-SharedTask-CD-SCO-test-cardboard.Preprocessed.csv"
)

# df_vectors = pd.DataFrame.from_dict(dict_vectors)
# print(df_vectors.head())
# dict_vectors["'"]

df_dev["Token_vector"] = df_dev.apply(lambda x: dict_vectors[x["Token"]], axis=1)
# df_dev["trigam_list_vectors"] = df_dev.apply(
#     lambda x: [dict_vectors[token] for token in x["trigram_list"]], axis=1
# )


df_dev.to_csv(
    r"D:\Studie\Business Analytics\Applied Text Mining\assignment3\Data\test\SEM-2012-SharedTask-CD-SCO-test-cardboard.Preprocessed.vectorized.csv",
    index=False,
)


# dict_vectors['"']

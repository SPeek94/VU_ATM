import pandas as pd
import numpy as np
from pathlib import Path

data_folder = Path("D:/Studie/Business Analytics/Applied Text Mining/assignment3/Data")

train_data = data_folder / "tmtrain.Preprocessed.pickle"


df_train = pd.read_pickle(train_data)

df_train.columns

vector_features = [
    "Token_vector",
    "next_token_vector",
    "prev_token_vector",
    "trigram_list_vectors",
    "prev_bigram_list_vectors",
    "next_bigram_list_vectors",
]

cat_features = ["POS", "POS_TAG", "Dependency_Head", "Dependency_Label", "HECT", "HECA"]

all_features = vector_features + cat_features

target = ["Negation_cue"]


def combine_vector_features(df, list_of_features):
    df = df[list_of_features].applymap(np.ravel)
    combined_vectors = df.apply(np.concatenate, axis=1)
    array_of_vector = np.array(combined_vectors.values.tolist())
    return array_of_vector


df_vectors = combine_vector_features(df_train, vector_features)


def make_sparse_features(df, list_of_features):
    pass


# first_tokens = df_train.iloc[0:5]
# first_tokens = first_tokens[vector_features]
# first_tokens
# first_tokens = first_tokens


# first_tokens.apply(np.concatenate)


# firs_input_vector = first_tokens.apply(np.concatenate, axis=1)
# test = np.array(firs_input_vector.values.tolist()).shape

# test[6]
# test2 = firs_input_vector.reset_index().to_numpy()


# np.reshape(firs_input_vector, (len(firs_input_vector),-1))

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

data_folder = Path("D:/Studie/Business Analytics/Applied Text Mining/assignment3/Data")

train_data = data_folder / "tmtrain.Preprocessed.pickle"
dev_data = data_folder / "tmdev.Preprocessed.pickle"


df_train = pd.read_pickle(train_data)
df_test = pd.read_pickle(dev_data)

all_features = [
    "Token_vector",
    "next_token_vector",
    "prev_token_vector",
    "trigram_list_vectors",
    "prev_bigram_list_vectors",
    "next_bigram_list_vectors",
    "POS",
    "POS_TAG",
    "Dependency_Head",
    "Dependency_Label",
    "HECT",
    "HECA",
]


vector_features = [
    "Token_vector",
    "next_token_vector",
    "prev_token_vector",
    "trigram_list_vectors",
    "prev_bigram_list_vectors",
    "next_bigram_list_vectors",
]

cat_features = ["POS", "POS_TAG", "Dependency_Head", "Dependency_Label", "HECT", "HECA"]


target = ["Negation_cue"]


def combine_vector_features(df, list_of_features):
    df = df[list_of_features].applymap(np.ravel)
    combined_vectors = df.apply(np.concatenate, axis=1)
    array_of_vector = np.array(combined_vectors.values.tolist())
    return array_of_vector


def make_oneHot_features(df, list_of_features):
    df = df[list_of_features]
    one_hot_dummys = np.array(pd.get_dummies(df))
    return one_hot_dummys


arr_vectors_train = combine_vector_features(df_train, vector_features)
arr_cat_train = make_oneHot_features(df_train, cat_features)

x_train = np.concatenate((arr_vectors_train, arr_cat_train), axis=1)
y_train = df_train[target].values.reshape(
    -1,
)

arr_vectors_test = combine_vector_features(df_test, vector_features)
arr_cat_test = make_oneHot_features(df_test, cat_features)

x_test = np.concatenate((arr_vectors_test, arr_cat_test), axis=1)
y_test = df_test[target].values.reshape(
    -1,
)


clf = MLPClassifier(solver="adam", alpha=1e-5, hidden_layer_sizes=5, random_state=1)
clf.fit(x_train, y_train)


prediction = clf.predict(x_test)


results = classification_report(y_test, prediction)

print(results)

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from functools import reduce


df_test_circle = pd.read_pickle(
    r"D:\Studie\Business Analytics\Applied Text Mining\assignment3\Data\test\SEM-2012-SharedTask-CD-SCO-test-circle.Preprocessed.pickle"
)

df_test_cardboad = pd.read_pickle(
    r"D:\Studie\Business Analytics\Applied Text Mining\assignment3\Data\test\SEM-2012-SharedTask-CD-SCO-test-cardboard.Preprocessed.pickle"
)
df_train = pd.read_pickle(
    r"D:\Studie\Business Analytics\Applied Text Mining\assignment3\Data\tmtrain.Preprocessed.pickle"
)
df_dev = pd.read_pickle(
    r"D:\Studie\Business Analytics\Applied Text Mining\assignment3\Data\tmdev.Preprocessed.pickle"
)

all_features = [
    "Token_vector",  # 300
    "next_token_vector",  # 300
    "prev_token_vector",  # 300
    "trigram_list_vectors",  # [300,300,300] --> np.ravel() --> [900]
    "prev_bigram_list_vectors",  # 600  [ [prev], [token]  ] --> [[1,...,3] [5,...,8]] --> [1,...,3,5,...,8] [..,3,5,..]
    "next_bigram_list_vectors",  # 600
    "POS",
    "POS_TAG",
    "Dependency_Head",
    "Dependency_Label",
    "HECT",
    "HECA",
]

vect_features = [
    "Token_vector",
    "next_token_vector",
    "prev_token_vector",
    "trigram_list_vectors",
    "prev_bigram_list_vectors",
    "next_bigram_list_vectors",
]

bool_features = ["HECT", "HECA"]
cat_features = ["POS", "POS_TAG", "Dependency_Head", "Dependency_Label"]


enc = OneHotEncoder(handle_unknown="ignore")
enc.fit(df_train[cat_features])


def multiply_arrays(arrays):
    if isinstance(arrays, np.ndarray) and len(arrays) == 300:
        return arrays
    else:
        return np.prod(np.vstack(arrays), axis=0)


def sum_arrays(arrays):
    if isinstance(arrays, np.ndarray) and len(arrays) == 300:
        return arrays
    else:
        return np.sum(np.vstack(arrays), axis=0)


def combine_vector_features(df, list_of_features):
    df = df[list_of_features].applymap(sum_arrays)
    combined_vectors = df.apply(np.concatenate, axis=1)
    array_of_vector = np.array(combined_vectors.values.tolist())
    return array_of_vector


def make_oneHot_features(df, list_of_features):
    df = df[list_of_features]
    # one_hot_dummys = np.array(pd.get_dummies(df))
    one_hot_labels = enc.transform(df).toarray()
    return one_hot_labels


def combine_features(df, all_features):
    vector_features = [
        x for x in all_features if x not in set(cat_features + bool_features)
    ]
    print(f"list vector_features: {vector_features} \n\n")
    categorical_features = [
        x for x in all_features if x not in set(vect_features + bool_features)
    ]
    print(f"list categorical_features:{categorical_features} \n\n")
    boolean_features = [
        x for x in all_features if x not in set(vect_features + cat_features)
    ]
    print(f"list boolean_features:{boolean_features} \n\n")

    x_vector = combine_vector_features(df, vector_features)
    x_cat = make_oneHot_features(df, categorical_features)
    x_num = df[boolean_features].to_numpy()
    print(
        f"Shape of vector array: {x_vector.shape} \nShape of oneHot array {x_cat.shape} \nShape of num array {x_num.shape}"
    )
    x = np.concatenate((x_vector, x_cat, x_num), axis=1)
    return x


df_test_circle_cat = df_test_circle[cat_features]
df_test_cardboad_cat = df_test_cardboad[cat_features]
df_train_cat = df_train[cat_features]
df_dev_cat = df_dev[cat_features]

df_test_circle_cat = df_test_circle_cat.agg(["nunique"])
df_test_cardboad_cat = df_test_cardboad_cat.agg(["nunique"])
df_train_cat = df_train_cat.agg(["nunique"])
df_dev_cat = df_dev_cat.agg(["nunique"])


print(
    f"Sum of unique features df_test_circle_cat: {df_test_circle_cat.sum(axis=1)} \n \n {df_test_circle_cat} \n \n"
)
print(
    f"Sum of unique features df_test_cardboad_cat: {df_test_cardboad_cat.sum(axis=1)} \n  \n {df_test_cardboad_cat} \n \n"
)
print(
    f"Sum of unique features df_train_cat: {df_train_cat.sum(axis=1)} \n  \n {df_train_cat} \n  \n "
)
print(
    f"Sum of unique features df_dev_cat: {df_dev_cat.sum(axis=1)} \n  \n {df_dev_cat} \n  \n  \n "
)


df_test_circle_vec = df_test_circle[vect_features]
df_test_cardboad_vec = df_test_cardboad[vect_features]
df_train_vec = df_train[vect_features]
df_dev_vec = df_dev[vect_features]

df_test_circle_vec = df_test_circle_vec.applymap(np.ravel)
df_test_cardboad_vec = df_test_cardboad_vec.applymap(np.ravel)
df_train_vec = df_train_vec.applymap(np.ravel)
df_dev_vec = df_dev_vec.applymap(np.ravel)


print("x_test_circle")
x_test_circle = combine_features(df_test_circle, all_features)
print("---------------------\n")

print("x_test_cardboard")
x_test_cardboard = combine_features(df_test_cardboad_vec, all_features)
print("---------------------\n")

print("x_train")
x_train = combine_features(df_train, all_features)
print("---------------------\n")

print("x_dev")
x_train = combine_features(df_dev, all_features)
print("---------------------\n")

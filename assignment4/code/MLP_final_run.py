import pandas as pd
import numpy as np
from pathlib import Path
from copy import deepcopy
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
import keras.backend as K
import keras
import pickle
from kerashypetune import KerasBayesianSearch, KerasBayesianSearchCV
from hyperopt import hp, Trials

data_folder = Path("D:/Studie/Business Analytics/Applied Text Mining/assignment3/Data")

train_data = data_folder / "tmtrain.Preprocessed.pickle"
dev_data = data_folder / "tmdev.Preprocessed.pickle"
test_data_circle = (
    data_folder / "test" / "SEM-2012-SharedTask-CD-SCO-test-circle.Preprocessed.pickle"
)
test_data_cardboard = (
    data_folder
    / "test"
    / "SEM-2012-SharedTask-CD-SCO-test-cardboard.Preprocessed.pickle"
)


df_train = pd.read_pickle(train_data)
df_dev = pd.read_pickle(dev_data)
df_test_circle = pd.read_pickle(test_data_circle)
df_test_cardboard = pd.read_pickle(test_data_cardboard)

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

features_after_ablations = [
    "prev_bigram_list_vectors",  # 600  [ [prev], [token]  ] --> [[1,...,3] [5,...,8]] --> [1,...,3,5,...,8] [..,3,5,..]
    "next_bigram_list_vectors",  # 600
    "POS",
    "POS_TAG",
    "HECT",
]

vect_features = [
    "prev_bigram_list_vectors",
    "next_bigram_list_vectors",
]

cat_features = ["POS", "POS_TAG"]
bool_features = ["HECT"]

target = ["Negation_cue"]


enc_features = OneHotEncoder(handle_unknown="ignore")
enc_features.fit(df_train[cat_features])

enc_labels = OneHotEncoder(handle_unknown="ignore")
enc_labels.fit(df_train[target])
target_labels = enc_labels.get_feature_names_out(target)


def sum_arrays(arrays):
    """
    Function to sum a list of arrays

    input: list of arrays with exact same shape

    output: single array with the same shape as on of the arrays in the list
    """
    if isinstance(arrays, np.ndarray) and len(arrays) == 300:
        return arrays
    else:
        return np.sum(np.vstack(arrays), axis=0)


def combine_vector_features(df, list_of_features):
    """
    Function that combines a list of vectors to one single vector

    input = dataframe and list with all vectorized features to be combined

    output = An array og the combined features
    """
    df = df[list_of_features].applymap(sum_arrays)
    combined_vectors = df.apply(np.concatenate, axis=1)
    array_of_vector = np.array(combined_vectors.values.tolist())
    return array_of_vector


def make_oneHot_features(df, list_of_features, enc_features):
    """
    Function to create on hot features

    input: dataframe and list with all categorical features to be encoded and the encoder

    output: an array containing all one hot encoded features
    """
    df = df[list_of_features]
    one_hot_labels = enc_features.transform(df).toarray()
    return one_hot_labels


def combine_features(df, all_features, enc_features):
    """
    Function to combine vectorized features and one hot encoded features

    input: dataframe and list with all features to be combined and the encoder

    output: an array containing all instances with encoded and vectorized features combined to a single vector
    """
    vector_features = [
        x for x in all_features if x not in set(cat_features + bool_features)
    ]
    categorical_features = [
        x for x in all_features if x not in set(vect_features + bool_features)
    ]
    boolean_features = [
        x for x in all_features if x not in set(vect_features + cat_features)
    ]

    x_vector = combine_vector_features(df, vector_features)
    x_cat = make_oneHot_features(df, categorical_features, enc_features)
    x_num = df[boolean_features].to_numpy()

    x = np.concatenate((x_vector, x_cat, x_num), axis=1)
    return x


def get_f1(y_true, y_pred):
    """
    Function to get f1 score

    input: true labels and predicted labels

    out: F1 score
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


x_train = combine_features(df_train, features_after_ablations, enc_features)
y_train = enc_labels.transform(df_train[target]).toarray()

x_dev = combine_features(df_dev, features_after_ablations, enc_features)
y_dev = enc_labels.transform(df_dev[target]).toarray()

x_test_circle = combine_features(df_test_circle, features_after_ablations, enc_features)
y_test_circle = enc_labels.transform(df_test_circle[target]).toarray()


x_test_cardboard = combine_features(
    df_test_cardboard, features_after_ablations, enc_features
)
y_test_cardboard = enc_labels.transform(df_test_cardboard[target]).toarray()

x_test = np.vstack((x_test_circle, x_test_cardboard))
y_test = np.vstack((y_test_circle, y_test_cardboard))


def mlp(input_):
    """
    Compiles a sequential MLP model

    input: training data to determine shape

    output: compiled model
    """
    model = Sequential()
    model.add(
        Dense(
            260,
            activation="relu",
            kernel_initializer="random_normal",
            input_shape=(input_.shape[1],),
        )
    )
    model.add(Dropout(0.018676167324734698))
    model.add(Dense(157, kernel_initializer="random_normal", activation="relu"))
    model.add(Dropout(0.018676167324734698))
    model.add(Dense(3, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.0012276333208346736),
        metrics=[get_f1],
    )
    model.summary()
    return model


model = mlp(x_train)

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_dev, y_dev),
    epochs=5,
    batch_size=128,
)

loss_dev, F1_score_dev = model.evaluate(x_dev, y_dev)
loss_test, F1_score_test = model.evaluate(x_test, y_test)


predictions_dev = model.predict(x_dev).round()
pred_labels_dev = enc_labels.inverse_transform(predictions_dev).ravel()


predictions_test = model.predict(x_test).round()
pred_labels_test = enc_labels.inverse_transform(predictions_test).ravel()

col_to_keep = ["Token", "trigram", "idx_sent", "HECT", "HECA", "Negation_cue"]


df_results_test = pd.concat(
    [df_test_circle[col_to_keep], df_test_cardboard[col_to_keep]], ignore_index=True
)
df_results_test["pred_MLP"] = pred_labels_test

df_results_dev = df_dev[col_to_keep]
df_results_dev["pred_MLP"] = pred_labels_dev

df_results_test.to_csv(
    r"D:\Studie\Business Analytics\Applied Text Mining\assignment5\resuts\MLP_test_result.csv",
    index=False,
)
df_results_dev.to_csv(
    r"D:\Studie\Business Analytics\Applied Text Mining\assignment5\resuts\MLP_dev_result.csv",
    index=False,
)

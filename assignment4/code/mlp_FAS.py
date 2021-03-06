import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPClassifier
import sklearn
from copy import deepcopy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from functools import reduce
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
import tensorflow.keras
import pickle

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
df_val = pd.read_pickle(dev_data)
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

vect_features = [
    "Token_vector",
    "next_token_vector",
    "prev_token_vector",
    "trigram_list_vectors",
    "prev_bigram_list_vectors",
    "next_bigram_list_vectors",
]

cat_features = ["POS", "POS_TAG", "Dependency_Head", "Dependency_Label"]
bool_features = ["HECT", "HECA"]

target = ["Negation_cue"]


enc_labels = OneHotEncoder(handle_unknown="ignore")
enc_labels.fit(df_train[target])
target_labels = enc_labels.get_feature_names_out(target)


def multiply_arrays(arrays):
    """
    Function to multiplay a list of arrays

    input: list of arrays with exact same shape

    output: single array with the same shape as on of the arrays in the list
    """
    if isinstance(arrays, np.ndarray) and len(arrays) == 300:
        return arrays
    else:
        return np.prod(np.vstack(arrays), axis=0)


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


def mlp(input_):
    """
    Compiles a sequential MLP model

    input: training data to determine shape

    output: compiled model
    """
    model = Sequential()
    model.add(
        Dense(
            512,
            activation="relu",
            kernel_initializer="glorot_uniform",
            input_shape=(input_.shape[1],),
        )
    )
    model.add(Dropout(0.5))
    model.add(Dense(128, kernel_initializer="glorot_uniform", activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=[get_f1])
    model.summary()
    return model


def feature_ablation(df_train, df_val, all_features):
    """
    Function that removes a single feature and test the impact of that removal, feature with least impact will be deleted.

    input: train and dev set and list of all features

    output: list containing the best score per run and order of removal
    """

    best_scores = []
    order_of_removal = []
    features_to_use = all_features.copy()

    while len(features_to_use) > 1:

        F1_scores = []
        print(
            f"\nFeatures used for training ({len(features_to_use)}): {features_to_use}\n \n"
        )

        for feature in features_to_use:

            print(f"Feature removed for this model run: {feature}\n")

            temp_features = deepcopy(features_to_use)
            temp_features.remove(feature)

            cat_features_ablation = [
                x for x in temp_features if x not in set(vect_features + bool_features)
            ]

            enc_features = OneHotEncoder(handle_unknown="ignore")
            enc_features.fit(df_train[cat_features_ablation])

            x_train = combine_features(df_train, temp_features, enc_features)
            y_train = enc_labels.transform(df_train[target]).toarray()

            x_val = combine_features(df_val, temp_features, enc_features)
            y_val = enc_labels.transform(df_val[target]).toarray()

            x_train_val = np.vstack((x_train, x_val))
            y_train_val = np.vstack((y_train, y_val))

            n_data = x_train_val.shape[0]
            n_samples = 20000
            idx = np.random.randint(n_data, size=n_samples)
            x_test = x_train_val[idx, :]
            y_test = y_train_val[idx, :]

            model = mlp(x_train)

            history = model.fit(
                x_train,
                y_train,
                validation_data=(x_val, y_val),
                epochs=5,
                batch_size=128,
            )

            loss, F1_score = model.evaluate(x_test, y_test)

            F1_scores.append(F1_score)
            print(f"F1 score: {F1_score}\n")

        best_scores.append(np.max(F1_scores))
        index_worst_feature = np.argmax(F1_scores)
        feature_to_delete = features_to_use[index_worst_feature]
        print(
            f"\nThe feature that had the least impact and therefore removed is: {feature_to_delete}\n"
        )

        order_of_removal.append(feature_to_delete)
        features_to_use.remove(feature_to_delete)

    return order_of_removal, best_scores


order_of_removal, best_scores = feature_ablation(
    df_train, df_val, ["POS_TAG", "next_bigram_list_vectors"]
)

with open("order_of_removal.pickle", "wb") as handle:
    pickle.dump(order_of_removal, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("best_scores.pickle", "wb") as handle:
    pickle.dump(best_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

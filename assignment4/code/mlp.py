import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from functools import reduce
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
import keras.backend as K
import keras

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

enc_features = OneHotEncoder(handle_unknown="ignore")
enc_features.fit(df_train[cat_features])
enc_features.get_feature_names_out(cat_features)

enc_labels = OneHotEncoder(handle_unknown="ignore")
enc_labels.fit(df_train[target])
target_labels = enc_labels.get_feature_names_out(target)


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
    one_hot_labels = enc_features.transform(df).toarray()
    return one_hot_labels


def combine_features(df, all_features):
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
    x_cat = make_oneHot_features(df, categorical_features)
    x_num = df[boolean_features].to_numpy()

    # print(
    #     f"Shape of vector array: {x_vector.shape} \nShape of oneHot array {x_cat.shape} \nShape of num array {x_num.shape}"
    # )

    x = np.concatenate((x_vector, x_cat, x_num), axis=1)
    return x


x_train = combine_features(df_train, all_features)
y_train = enc_labels.transform(df_train[target]).toarray()
# y_train = df_train[target].values.reshape(
#     -1,
# )

x_val = combine_features(df_val, all_features)
y_val = enc_labels.transform(df_val[target]).toarray()
# y_val = df_val[target].values.reshape(
#     -1,
# )

x_test_circle = combine_features(df_test_circle, all_features)
y_test_circle = enc_labels.transform(df_test_circle[target]).toarray()
# y_test_circle = df_test_circle[target].values.reshape(
#     -1,
# )
# y_test_circle_oneHot = pd.get_dummies(y_test_circle)

x_test_cardboard = combine_features(df_test_cardboard, all_features)
y_test_cardboard = enc_labels.transform(df_test_cardboard[target]).toarray()
# y_test_cardboard = df_test_cardboard[target].values.reshape(
#     -1,
# )
# y_test_cardboard_oneHot = pd.get_dummies(y_test_cardboard)


# clf = MLPClassifier(solver="adam", alpha=1e-5, hidden_layer_sizes=5, random_state=1)
# clf.fit(x_train, y_train)


# prediction = clf.predict(x_test)


# results = classification_report(y_test, prediction)

# print(results)


def get_f1(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


# input_shape = x_train.shape[1]
# y_train_oneHot = pd.get_dummies(y_train)
# y_val_oneHot = pd.get_dummies(y_val)


model = Sequential()
model.add(
    Dense(
        512,
        activation="relu",
        kernel_initializer="glorot_uniform",
        input_shape=(x_train.shape[1],),
    )
)
model.add(Dropout(0.5))
model.add(Dense(128, kernel_initializer="glorot_uniform", activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(3, activation="softmax"))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=[get_f1])
model.summary()

history = model.fit(
    x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=128
)

score = model.evaluate(x_test_cardboard, y_test_cardboard)

loss, F1_score = score

predictions = np.argmax(model.predict(x_test_cardboard), axis=1)
labels = np.argmax(y_test_cardboard, axis=1)

pred = model.predict(x_test_cardboard).round()
pred = enc_labels.inverse_transform(pred)
pred = pred.flatten()
gold_labels = enc_labels.inverse_transform(y_test_cardboard)
gold_labels = gold_labels.flatten()


# pred = model.predict(x_test_cardboard).round()
# pred = pd.DataFrame(pred, columns=target_labels)
# gold_labels = pd.DataFrame(y_test_cardboard, columns=target_labels)


import pandas as pd
from collections import Counter
import numpy as np


def EvaluatePredictions(Predictions, Gold):
    """

    :param Predictions:
    :param Gold:
    :return:
    """

    Total = len(Gold)
    labcount = Counter(Gold)
    fcontribution = []
    macrof = []

    ClassInfo = []
    for label in set(Gold):
        contribution = labcount[label] / Total

        confdict = {"TP": 0.1, "TN": 0.1, "FP": 0.1, "FN": 0.1}
        scoredict = {"Precision": 0, "F_Score": 0, "Recall": 0}

        for i in range(Total):
            if Predictions[i] == Gold[i]:
                if Predictions[i] == label:
                    confdict["TP"] += 1
                else:
                    confdict["TN"] += 1

            else:
                if Predictions[i] == label:
                    confdict["FP"] += 1
                else:
                    confdict["FN"] += 1

        # Failures = (confdict['FP'] + confdict['FN'], confdict['FP'], confdict['FN'])
        # Accuracy = (confdict['TP'] + confdict['TN']) / Total
        scoredict["Precision"] = round(
            confdict["TP"] / (confdict["TP"] + confdict["FP"]), 4
        )
        scoredict["Recall"] = round(
            confdict["TP"] / (confdict["TP"] + confdict["FN"]), 4
        )
        scoredict["F_Score"] = round(
            (2 * (scoredict["Precision"] * scoredict["Recall"]))
            / (scoredict["Precision"] + scoredict["Recall"]),
            4,
        )
        fcontribution.append((contribution * scoredict["F_Score"]))
        macrof.append(scoredict["F_Score"])

        print(label, ": Scores:", scoredict, "\n", "Confusions:", confdict, "\n")

        ClassInfo.append((scoredict, confdict))

    Avg_F_Scores = {
        "Micro_F_Score": np.sum(fcontribution),
        "Macro_F_Score": np.mean(macrof),
    }

    print(Avg_F_Scores)

    return confdict, Avg_F_Scores


EvaluatePredictions(pred, gold_labels)

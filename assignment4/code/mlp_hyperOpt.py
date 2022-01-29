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


with open(
    r"D:\Studie\Business Analytics\Applied Text Mining\order_of_removal.pickle", "rb"
) as handle:
    order_of_removal = pickle.load(handle)

with open(
    r"D:\Studie\Business Analytics\Applied Text Mining\best_scores.pickle", "rb"
) as handle:
    best_scores = pickle.load(handle)


data_folder = Path("D:/Studie/Business Analytics/Applied Text Mining/assignment3/Data")

train_data = data_folder / "tmtrain.Preprocessed.pickle"
dev_data = data_folder / "tmdev.Preprocessed.pickle"

df_train = pd.read_pickle(train_data)
df_dev = pd.read_pickle(dev_data)


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
    if isinstance(arrays, np.ndarray) and len(arrays) == 300:
        return arrays
    else:
        return np.sum(np.vstack(arrays), axis=0)


def combine_vector_features(df, list_of_features):
    df = df[list_of_features].applymap(sum_arrays)
    combined_vectors = df.apply(np.concatenate, axis=1)
    array_of_vector = np.array(combined_vectors.values.tolist())
    return array_of_vector


def make_oneHot_features(df, list_of_features, enc_features):
    df = df[list_of_features]
    one_hot_labels = enc_features.transform(df).toarray()
    return one_hot_labels


def combine_features(df, all_features, enc_features):
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


def get_f1(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def mlp(input_):
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


def mlpv2(param, input_):
    model = Sequential()
    model.add(
        Dense(
            param["layer1"],
            kernel_initializer=param["init"],
            activation=param["activ1"],
            input_shape=(input_.shape[1],),
        )
    )
    model.add(Dropout(param["dropout"]))
    model.add(
        Dense(
            param["layer2"],
            kernel_initializer=param["init"],
            activation=param["activ2"],
            input_shape=(input_.shape[1],),
        )
    )
    model.add(Dropout(param["dropout"]))
    model.add(Dense(3, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=param["lr"]),
        metrics=[get_f1],
    )
    model.summary()
    return model


x_train = combine_features(df_train, features_after_ablations, enc_features)
y_train = enc_labels.transform(df_train[target]).toarray()

x_dev = combine_features(df_dev, features_after_ablations, enc_features)
y_dev = enc_labels.transform(df_dev[target]).toarray()

x_train_dev = np.vstack((x_train, x_dev))
y_train_dev = np.vstack((y_train, y_dev))

n_data = x_train_dev.shape[0]
n_samples = 20000
idx = np.random.randint(n_data, size=n_samples)
x_test = x_train_dev[idx, :]
y_test = y_train_dev[idx, :]


param_grid = {
    "layer1": 128 + hp.randint("layer1", 512),
    "layer2": 64 + hp.randint("layer2", 256),
    "dropout": hp.uniform("dropout", 0, 0.5),
    "lr": hp.loguniform("lr", np.log(0.001), np.log(0.02)),
    "activ1": hp.choice("activ1", ["elu", "relu", "sigmoid"]),
    "activ2": hp.choice("activ2", ["elu", "relu", "sigmoid"]),
    "init": hp.choice(
        "init", ["glorot_uniform", "glorot_normal", "random_normal", "random_uniform"]
    ),
    "epochs": 5,
    "batch_size": hp.choice("batch_size", [32, 64, 128, 256]),
}


cv = KFold(n_splits=5, shuffle=True)

get_model = lambda x: mlpv2(param=x, input_=x_train_dev)

kbs = KerasBayesianSearchCV(
    get_model, param_grid, cv=cv, monitor="get_f1", greater_is_better=True, n_iter=100
)

kbs.search(x_train_dev, y_train_dev, trials=Trials())

scores = kbs.folds_scores
print(f"All F1 scores during hyperOpt: {scores}\n")
best_score = kbs.folds_best_score
print(f"Best F1 score optained during hyperOpt: {best_score}\n")
best_params = kbs.folds_best_params
print(f"Best parameters for training: {best_params}\n")
best_model = kbs.folds_best_models
print(f"Best model obtained dureing hyperOpt: {best_model}\n")

hyperOpt_results = (scores, best_score, best_params)
for fold in best_model:
    best_model[fold].save(
        fr"D:\Studie\Business Analytics\Applied Text Mining\assignment4\models\best_model_{fold}"
    )


with open(
    r"D:\Studie\Business Analytics\Applied Text Mining\assignment4\models\MLP_results_after_hyperOpt.pickle",
    "wb",
) as handle:
    pickle.dump(hyperOpt_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

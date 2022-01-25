import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import sklearn_crfsuite
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from Evaluation import EvaluatePredictions

enc = OneHotEncoder(sparse=False, handle_unknown="ignore")


def extract_features_and_labels(
    trainingfile, testfile, featureselection=["POS", "Highly Expected Cue"]
):

    print("Extracting")

    features_df = pd.read_csv(trainingfile, encoding="utf-8")
    test_features_df = pd.read_csv(testfile, encoding="utf-8")

    test_features = []
    for i in range(len(test_features_df)):
        # test_features.append([test_features_df['POS'][i], test_features_df['Highly Expected Cue'][i]])
        test_features.append([test_features_df[fea][i] for fea in featureselection])
    enc.fit(test_features)
    test_features = enc.transform(test_features)

    training_features = []
    for i in range(len(features_df)):
        training_features.append([features_df[fea][i] for fea in featureselection])
    training_features = enc.transform(training_features)

    targets = list(features_df["Negation_cue"])
    test_targets = list(test_features_df["Negation_cue"])

    print("Done")

    return training_features, test_features, targets, test_targets, test_features_df


def create_classifier(classifier):
    """
    :param classifier:
    :return:
    """

    if classifier == "SVM":
        clf = svm.SVC(max_iter=10000)
    if classifier == "CRF":
        clf = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True,
        )
    if classifier == "LogReg":
        clf = LogisticRegression(random_state=0)

    return clf


def classifydata(trainingfile, testfile, feasel, model):
    """
    :param trainingfile:
    :param testfile:
    :param feasel:
    :param classifier:
    :return:
    """
    classifier = create_classifier(model)

    numfeas = len(feasel)
    print(numfeas)

    X, X_test, y, gold, Features_df = extract_features_and_labels(
        trainingfile, testfile, feasel
    )
    # if model == 'CRF':
    # print(type(X[0]))
    # print(y[0])
    print("fitting")
    classifier.fit(X, y)
    print("Done")
    predictions = classifier.predict(X_test)
    pred = list(predictions)

    # Create File with predictions
    Features_df["Prediction"] = predictions
    outfile = testfile.replace(
        os.path.basename(testfile),
        "Predictions/" + model + "_" + "-".join(feature_selection) + ".csv",
    )
    Features_df.to_csv(outfile)
    print("Predictions File Created")

    return pred, gold  # , outfile


def main(trainingfile, testfile, FeaSel, models):
    for model in models:
        pred, gold = classifydata(trainingfile, testfile, FeaSel, model)
        print("Done")
        print("Evaluating")
        EvaluatePredictions(pred, gold)
        print("Done")


train_csv = "/Users/joshuawork/Desktop/Seb_Neg_Cue_Det/Data/tmtrain.Preprocessed.csv"
dev_csv = "/Users/joshuawork/Desktop/Seb_Neg_Cue_Det/Data/tmdev.Preprocessed.csv"
feature_selection = ["HECT", "HECA", "POS_TAG"]
main(train_csv, dev_csv, feature_selection, ["LogReg"])

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
        contribution = labcount[label]/Total

        confdict = {'TP': 0.1, 'TN': 0.1, 'FP': 0.1, 'FN': 0.1}
        scoredict = {'Precision': 0, 'F_Score': 0, 'Recall': 0}

        for i in range(Total):
            if Predictions[i] == Gold[i]:
                if Predictions[i] == label:
                    confdict['TP'] += 1
                else:
                    confdict['TN'] += 1

            else:
                if Predictions[i] == label:
                    confdict['FP'] += 1
                else:
                    confdict['FN'] += 1

        #Failures = (confdict['FP'] + confdict['FN'], confdict['FP'], confdict['FN'])
        #Accuracy = (confdict['TP'] + confdict['TN']) / Total
        scoredict['Precision'] = round(confdict['TP'] / (confdict['TP'] + confdict['FP']), 4)
        scoredict['Recall'] = round(confdict['TP'] / (confdict['TP'] + confdict['FN']),4)
        scoredict['F_Score'] = round((2 * (scoredict['Precision'] * scoredict['Recall'])) / (scoredict['Precision'] + scoredict['Recall']), 4)
        fcontribution.append((contribution * scoredict['F_Score']))
        macrof.append(scoredict['F_Score'])

        print(label, ': Scores:', scoredict, '\n', 'Confusions:', confdict, '\n')

        ClassInfo.append((scoredict, confdict))

    Avg_F_Scores = {'Micro_F_Score': np.sum(fcontribution), 'Macro_F_Score' : np.mean(macrof)}

    print(Avg_F_Scores)

    return confdict, Avg_F_Scores

def main(Predfile):
    preds_df = pd.read_csv(Predfile, encoding="utf-8")
    Predictions = preds_df['Prediction']
    Gold = preds_df['Negation_cue']
    EvaluatePredictions(Predictions, Gold)


# testev = "../Data/Predictions/LogReg_HECT-HECA-POS_TAG.csv"
# main(testev)
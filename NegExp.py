import pandas as pd
from collections import Counter
import re
import numpy as np

# train = "/Users/joshuawork/Desktop/Seb_Neg_Cue_Det/train/SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt"
#
# df_train = pd.read_csv(
#     train,
#     header=None,
#     encoding="utf-8",
#     sep="\t",
#     index_col=False,
# )
#
# df_train.columns = ["Chapter", "Sentence_ID", "Token_ID", "Token", "Negation_cue"]


#Code For Analysis of Cues
Negcues = []
for i in range(len(df_train)):
    if df_train['Negation_cue'][i] != 'O':
        Negcues.append(df_train['Token'][i].lower())
CueCount = Counter(Negcues)
Frequent = [tuple[0] for tuple in CueCount.most_common(8)]
FrequentSet = set(Frequent)
InfrequentSet = set([token for token in Negcues if token not in Frequent])
Rest = set()

for cue in InfrequentSet:
    if not (cue.startswith(('un', 'in', 'dis', 'im', 'ir', 'no')) or cue.endswith(('less', 'lessly', 'lessness'))):
        Rest.add(cue)

# ExpCueBool = []
# for i, word in enumerate(df_output_parser["Word"]):
#     word = word.lower()
#     if word in ['in', 'less']:
#         ExpCueBool.append(0)
#     elif word.startswith(('un', 'in', 'dis')) or word.endswith(('less', 'lessly', 'lessness')):
#         ExpCueBool.append(1)
#     elif word.startswith(('im', 'ir', 'no')):
#         ExpCueBool.append(1)
#     else:
#         ExpCueBool.append(0)

#Code for Creating HighexpBoolean column

def CreateHighExpBool(dataframe, wordcolumn, POScolumn, FrequentSet = ("n't", 'never', 'no', 'none', 'nor', 'not', 'nothing', 'without')):
    PosVocab = []
    for i, word in enumerate(dataframe[wordcolumn]):
        if dataframe[POScolumn][i] in ['NOUN', 'ADV', 'VERB', 'ADJ', 'VBN']:
            PosVocab.append(word)

    HighExpCueBool = []
    for word in dataframe[wordcolumn]:
        word = word.lower()
        if word[:2] in ['im', 'ir', 'no', 'un', 'in'] and word[2:] in PosVocab:
            HighExpCueBool.append(1)
        elif word[:3] in ['dis'] and word[3:] in PosVocab:
            HighExpCueBool.append(1)
        elif word.endswith(('less', 'lessly', 'lessness')) and word not in ['unless','bless']: #and word[-4:] in Vocab:
            HighExpCueBool.append(1)
        elif word in FrequentSet:
            HighExpCueBool.append(1)
        else:
            HighExpCueBool.append(0)

    print(sum(HighExpCueBool))

    return HighExpCueBool

# HighExpCueBool = CreateHighExpBool(df_output_parser, "Word", 'POS', FrequentSet)
#
# #Add Column to DataFrame
# df_output_parser['Highly Expected Cue'] = HighExpCueBool
# df_output_parser.sort_values(by=['Highly Expected Cue'], inplace=True)

def AddHighExpBoolCol(dffile, outfile, wordcolumn = "Token", POScolumn = 'POS'):

    old_df = pd.read_csv(dffile, encoding="utf-8")
    HighExpCueBool = CreateHighExpBool(old_df, wordcolumn, POScolumn)
    old_df['Highly Expected Cue'] = HighExpCueBool
    new_df = old_df
    new_df.to_csv(outfile)

    return

AddHighExpBoolCol("/Users/joshuawork/Desktop/Seb_Neg_Cue_Det/train/TrainingFeaturesCSV", "/Users/joshuawork/Desktop/Seb_Neg_Cue_Det/train/TrainingFeaturesNeg.csv")
AddHighExpBoolCol("/Users/joshuawork/Desktop/Seb_Neg_Cue_Det/train/DevFeaturesCSV", "/Users/joshuawork/Desktop/Seb_Neg_Cue_Det/train/NegExpDF.csv")
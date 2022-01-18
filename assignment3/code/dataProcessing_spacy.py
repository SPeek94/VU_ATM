import spacy
import pandas as pd
import numpy as np

nlp = spacy.load("en_core_web_sm")

df_train = pd.read_csv(
    r"D:\Studie\Business Analytics\Applied Text Mining\assignment3\train\SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt",
    header=None,
    encoding="utf-8",
    sep="\t",
    index_col=False,
)

df_train.columns = ["Chapter", "Sentence_ID", "Token_ID", "Token", "Negation_cue"]

df_train["Sentence_ID_unique"] = df_train.groupby(["Chapter", "Sentence_ID"]).ngroup()

token_sent = []
for i in range(len(set(df_train["Sentence_ID_unique"]))):
    doc = nlp(" ".join(df_train[df_train.Sentence_ID_unique == i]["Token"]))
    token_sent.append(doc)


listOfDicts = []
sent_idx = 0
for doc in token_sent:  # for each tokenized/processed sentence in the list
    for (
        sent
    ) in doc.sents:  # take each sentence, since we only have 1 sentence it loops 1 time
        for i, word in enumerate(sent):  # go trough each token/word
            if word.head == word:  # reset counter
                head_idx = 0
            else:  # otherwise calculate idx
                head_idx = word.head.i - sent[0].i + 1
            dict_parser_output = (
                {}
            )  # make dictionary and fill it values, as showed in the report appendix II
            dict_parser_output["idx_sent"] = sent_idx
            dict_parser_output["Token#"] = i + 1
            dict_parser_output["Word"] = word.text
            dict_parser_output["Lemma"] = word.lemma_
            dict_parser_output["POS"] = word.pos_
            dict_parser_output["POS_TAG"] = word.tag_
            dict_parser_output["Dependency Head"] = head_idx
            dict_parser_output["Dependency Label"] = word.dep_
            listOfDicts.append(dict_parser_output)  # append to list
    sent_idx += 1

columns_ = [
    "Token#",
    "Word",
    "Lemma",
    "POS",
    "POS_TAG" "Dependency Head",
    "Dependency Label",
    "idx_sent",
]


df_output_parser = pd.DataFrame(listOfDicts, columns=columns_)
df_output_parser.head()  # shows results partly


"""
ngrams
previous/next token/pos
affixes
"""

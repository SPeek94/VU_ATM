import spacy
import pandas as pd
import numpy as np
import re
from spacy.tokenizer import Tokenizer

df_train = pd.read_csv(
    r"D:\Studie\Business Analytics\Applied Text Mining\assignment3\train\SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt",
    header=None,
    encoding="utf-8",
    sep="\t",
    index_col=False,
)


nlp = spacy.load("en_core_web_sm")

nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r"(?<=Chapter )\d+.").match)

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
            dict_parser_output["Token_ID"] = i + 1
            dict_parser_output["Token"] = word.text
            dict_parser_output["Lemma"] = word.lemma_
            dict_parser_output["POS"] = word.pos_
            dict_parser_output["POS_TAG"] = word.tag_
            dict_parser_output["Dependency_Head"] = head_idx
            dict_parser_output["Dependency_Label"] = word.dep_
            listOfDicts.append(dict_parser_output)  # append to list
    sent_idx += 1

columns_ = [
    "Token_ID",
    "Token",
    "Lemma",
    "POS",
    "POS_TAG",
    "Dependency_Head",
    "Dependency_Label",
    "idx_sent",
]


df_output_parser = pd.DataFrame(listOfDicts, columns=columns_)

df_output_parser["next"] = df_output_parser.Token.shift(fill_value="None")
df_output_parser["prev"] = df_output_parser.Token.shift(-1, fill_value="None")

df_output_parser.head(6)  # shows results partly
df_output_parser

"""
ngrams
previous/next token/pos
affixes
"""

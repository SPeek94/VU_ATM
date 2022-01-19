import spacy
import pandas as pd
import numpy as np
import re
from spacy.tokenizer import Tokenizer
from spacy.symbols import ORTH

df_train = pd.read_csv(
    r"D:\Studie\Business Analytics\Applied Text Mining\assignment3\train\SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt",
    header=None,
    encoding="utf-8",
    sep="\t",
    index_col=False,
)

from spacy.tokens import Doc


df_train.columns = ["Chapter", "Sentence_ID", "Token_ID", "Token", "Negation_cue"]

df_train["Sentence_ID_unique"] = df_train.groupby(["Chapter", "Sentence_ID"]).ngroup()


sentences = []
for i in range(len(set(df_train["Sentence_ID_unique"]))):
    doc = " ".join(df_train[df_train.Sentence_ID_unique == i]["Token"])
    sentences.append(doc)

sent = df_train[df_train["Sentence_ID_unique"] == 0]
sent_1 = list(df_train[df_train["Sentence_ID_unique"] == 0]["Token"])
sent_2 = list(df_train[df_train["Sentence_ID_unique"] == 1]["Token"])

sent_1
sent_2

nlp = spacy.load("en_core_web_sm")

doc = Doc(nlp.vocab, words=df_train["Token"])

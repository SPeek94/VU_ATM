import pandas as pd
import numpy as np
import fasttext.util

fasttext.util.download_model("en", if_exists="ignore")  # English
ft = fasttext.load_model("cc.en.300.bin")

df_train = pd.read_csv(
    r"D:\Studie\Business Analytics\Applied Text Mining\assignment3\train\pre_processed_train.csv",
)

num_sentences = list(set(df_train["idx_sent"]))

sentences = []
for sent_id in num_sentences:
    sentences.append(list(df_train[df_train.idx_sent == sent_id]["Token"]))

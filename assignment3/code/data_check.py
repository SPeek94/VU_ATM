import pandas as pd
import numpy as np

df_train = pd.read_pickle(
    r"D:\Studie\Business Analytics\Applied Text Mining\assignment3\Data\test\SEM-2012-SharedTask-CD-SCO-test-cardboard.Preprocessed.pickle"
)


# np.zeros((300,))

# first_token.next_token == "None"

# if first_token.next_token == "None":
#     first_token.next_token_vector = np.zeros((300,))
# if first_token.next_token == "None":
#     first_token.next_token_vector = np.zeros((300,))


# def vector_firs_last_token(df_train):
#     if df_train.next_token == "None":
#         df_train.next_token_vector = np.zeros((300,))
#     elif df_train.prev_token == "None":
#         df_train.prev_token_vector = np.zeros((300,))

#     return df_train


df_train["next_token_vector"].iloc[-1] = np.zeros((300,))
df_train["prev_token_vector"].iloc[0] = np.zeros((300,))

first_token = df_train.iloc[0]
last_token = df_train.iloc[-1]

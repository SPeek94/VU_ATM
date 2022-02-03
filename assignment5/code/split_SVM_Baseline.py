import pandas as pd
import numpy as np

df_SVM_dev = pd.read_csv(
    r"D:\Studie\Business Analytics\Applied Text Mining\assignment5\resuts\SVM_Base_dev_result.csv"
)
df_SVM_test = pd.read_csv(
    r"D:\Studie\Business Analytics\Applied Text Mining\assignment5\resuts\SVM_Base_test_result.csv"
)

df_base_dev = df_SVM_dev.copy()

df_base_dev = df_base_dev.drop(columns="pred_SVM")
df_SVM_dev = df_SVM_dev.drop(columns="pred_base")

df_base_test = df_SVM_test.copy()

df_base_test = df_base_test.drop(columns="pred_SVM")
df_SVM_test = df_SVM_test.drop(columns="pred_base")


df_base_dev.to_csv(
    r"D:\Studie\Business Analytics\Applied Text Mining\assignment5\resuts\Base_dev_result.csv",
    index=False,
)
df_base_test.to_csv(
    r"D:\Studie\Business Analytics\Applied Text Mining\assignment5\resuts\Base_test_result.csv",
    index=False,
)

df_SVM_dev.to_csv(
    r"D:\Studie\Business Analytics\Applied Text Mining\assignment5\resuts\SVM_dev_result.csv",
    index=False,
)
df_SVM_test.to_csv(
    r"D:\Studie\Business Analytics\Applied Text Mining\assignment5\resuts\SVM_test_result.csv",
    index=False,
)

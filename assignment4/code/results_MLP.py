import pandas as pd
import numpy as np
import pickle

with open(
    r"D:\Studie\Business Analytics\Applied Text Mining\assignment4\models\MLP_results_after_hyperOpt.pickle",
    "rb",
) as handle:
    hyperOpt_results = pickle.load(handle)

scores, best_scores, best_params = hyperOpt_results

best_params["fold 5"]

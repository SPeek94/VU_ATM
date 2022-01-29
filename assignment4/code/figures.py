import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle

with open(
    r"D:\Studie\Business Analytics\Applied Text Mining\order_of_removal.pickle", "rb"
) as handle:
    order_of_removal = pickle.load(handle)

with open(
    r"D:\Studie\Business Analytics\Applied Text Mining\best_scores.pickle", "rb"
) as handle:
    best_scores = pickle.load(handle)


order_of_removal.reverse()
order_of_removal.append("Use all features")

best_scores.reverse()
best_scores.append(best_scores[-1])


df_bar = pd.DataFrame(
    {"F1-score": best_scores}, columns=["F1-score"], index=order_of_removal
)

df_bar.plot.barh(
    figsize=(12, 6),
    legend=False,
    xlim=(np.min(best_scores) - 0.0001, np.max(best_scores) + 0.0001),
)

plt.title(
    "The effect of ablating the least important feature when running an MLP model"
)
plt.xlabel("F1-score", size=14)
plt.ylabel("Last ablated feature", size=14)
plt.savefig("FAS_MLP.pdf")
plt.show()

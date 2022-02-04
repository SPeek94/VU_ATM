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
best_scores

order_of_removal.append("next_bigram_list_vectors")
best_scores.append(0.9959984421730042)


# Next code is to create the feature ablation bar plot for the MLP model

df_bar = pd.DataFrame(
    {"F1-score": best_scores}, columns=["F1-score"], index=order_of_removal
)

ax = df_bar.plot.barh(
    figsize=(12, 6),
    legend=False,
    xlim=(np.min(best_scores) - 0.0001, np.max(best_scores) + 0.0001),
)

ax.text(best_scores[-2] + 0.000001, 10 - 0.06, order_of_removal[-2:], color="black")
for i in [1, 3, 4, 5, 6]:
    ax.text(
        best_scores[-2] + 0.000001, 12 - i - 0.06, order_of_removal[-i:], color="white"
    )

plt.title("The effect of removing the next least important feature (flipped)", size=16)
plt.xlabel("F1-score", size=14)
plt.ylabel("Last added feature", size=14)
plt.savefig("FAS_MLPV2.pdf", bbox_inches="tight")
plt.show()

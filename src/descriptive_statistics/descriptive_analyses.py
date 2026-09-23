# %% [markdown]
# # Descriptive analyses of SIRTUIN6

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

RAW_PATH = Path("../../data/raw/SIRTUIN6.csv")
FIGURES_PATH = Path("../../data/figures")
TARGET = "Class"

FIGURES_PATH.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(RAW_PATH)
features = df.drop(columns=TARGET).select_dtypes("number")

# %% [markdown]
# ## Descriptive statistics

# %%
summary = features.describe().T
summary["skew"] = features.skew()
summary["kurtosis"] = features.kurt()
summary["missing"] = features.isna().sum()
summary

# %%
counts = df[TARGET].value_counts()
pd.DataFrame({"count": counts, "share": counts / len(df)})

# %%
df.groupby(TARGET).agg(["mean", "std"]).T

# %% [markdown]
# ## Distributions

# %%
normality = pd.DataFrame(
    {col: stats.shapiro(features[col].dropna()) for col in features},
    index=["statistic", "p_value"],
).T
normality["normal"] = normality["p_value"] > 0.05
normality

# %%
classes = [group for _, group in df.groupby(TARGET)]
class_comparison = pd.DataFrame(
    {col: stats.mannwhitneyu(classes[0][col], classes[1][col]) for col in features},
    index=["statistic", "p_value"],
).T
class_comparison

# %% [markdown]
# ## Figures

# %%
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, col in zip(axes.flat, features):
    sns.histplot(data=df, x=col, hue=TARGET, kde=True, ax=ax)
fig.tight_layout()
fig.savefig(FIGURES_PATH / "histograms.png", dpi=150)
plt.show()

# %%
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, col in zip(axes.flat, features):
    sns.boxplot(data=df, x=TARGET, y=col, ax=ax)
fig.tight_layout()
fig.savefig(FIGURES_PATH / "boxplots.png", dpi=150)
plt.show()

# %%
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(features.corr(), annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
fig.tight_layout()
fig.savefig(FIGURES_PATH / "correlations.png", dpi=150)
plt.show()

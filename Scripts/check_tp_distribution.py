import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv(
    "/Users/linustyden/Bachelor-Thesis-COSMO-Optimisation-in-C.-difficile/parameter_optimisation/raw_datasets/LHS grid/raw_dataset_regression.csv",
    sep=None, engine="python"
)

REGRESSORS = ["CDS_min", "IGR_min", "FD_CDS-CDS_min", "FD_IGR-CDS_min"]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, col in zip(axes.ravel(), REGRESSORS):
    ax.scatter(df[col], df["TP%"], alpha=0.4, s=15, edgecolors="none")
    ax.set_xlabel(col)
    ax.set_ylabel("TP%")
    ax.set_title(col)
plt.suptitle("Parameter vs TP% (raw data)", fontsize=12)
plt.tight_layout()
plt.savefig("parameter_optimisation/analysis/param_vs_tp_scatter.png", dpi=150)
plt.close()

print("Saved: parameter_optimisation/analysis/param_vs_tp_scatter.png")
print()
print("Unique TP% values:", sorted(df["TP%"].unique()))
print()
print(df["TP%"].describe())

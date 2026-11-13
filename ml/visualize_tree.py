import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.tree import plot_tree

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_FILE = PROJECT_DIR / "dataset" / "carbon_nmr_features.csv"
MODEL_FILE = PROJECT_DIR / "ml" / "decision_tree_model.joblib"

df = pd.read_csv(DATASET_FILE)

numeric_features = [
    "atomic_number",
    "degree",
    "hydrogen_count",
    "is_aromatic",
    "is_in_ring",
    "formal_charge",
    "oxygen_count",
    "nitrogen_count",
    "bond_order_sum"
]

categorical_features = ["hybridization"]

feature_names = numeric_features + [
    "hybridization_SP",
    "hybridization_SP2",
    "hybridization_SP3"
]

model = joblib.load(MODEL_FILE)

tree_model = model.named_steps["regressor"]

plt.figure(figsize=(20, 10))

plot_tree(
    tree_model,
    feature_names=feature_names,
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("Decision Tree Regression for 13C NMR")
plt.savefig(
    PROJECT_DIR / "ml" / "decision_tree.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Decision tree visualization saved successfully!")
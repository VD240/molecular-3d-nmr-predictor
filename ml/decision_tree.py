import pandas as pd
from pathlib import Path
import joblib

from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_FILE = (
    PROJECT_DIR / "dataset" / "carbon_nmr_features.csv"
)

MODEL_FILE = (
    PROJECT_DIR / "ml" / "decision_tree_model.joblib"
)

# Load dataset
df = pd.read_csv(DATASET_FILE)

print("Dataset loaded:", len(df), "carbon records")

# Features
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

X = df[numeric_features + categorical_features]
y = df["chemical_shift_ppm"]
groups = df["molecule_id"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", "passthrough", numeric_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ]
)

# Decision Tree Regression
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", DecisionTreeRegressor(
        max_depth=3,
        random_state=42
    ))
])

# Evaluation by molecule
number_of_molecules = groups.nunique()

if number_of_molecules < 2:
    raise ValueError("At least two molecules are required.")

cv = GroupKFold(n_splits=min(5, number_of_molecules))

predictions = cross_val_predict(
    model,
    X,
    y,
    groups=groups,
    cv=cv
)

mae = mean_absolute_error(y, predictions)
rmse = mean_squared_error(y, predictions) ** 0.5

print("\nDECISION TREE RESULTS")
print("-------------------------")
print("MAE:", round(mae, 3), "ppm")
print("RMSE:", round(rmse, 3), "ppm")

# Train final model
model.fit(X, y)

joblib.dump(model, MODEL_FILE)

print("\nDecision Tree trained successfully!")
print("Model saved:", MODEL_FILE)
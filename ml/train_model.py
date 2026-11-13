import pandas as pd
import joblib

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error

# PROJECT PATHS

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_FILE = (
    PROJECT_DIR / "dataset" / "carbon_nmr_features.csv"
)

MODEL_FILE = (
    PROJECT_DIR / "ml" / "carbon_nmr_model.joblib"
)

# LOAD DATASET

df = pd.read_csv(DATASET_FILE)

print("Dataset loaded successfully!")
print("Total carbon records:", len(df))

# SELECT FEATURES AND TARGET

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

categorical_features = [
    "hybridization"
]

target = "chemical_shift_ppm"

group_column = "molecule_id"

required_columns = (
    numeric_features
    + categorical_features
    + [target, group_column]
)

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )

X = df[numeric_features + categorical_features].copy()

y = pd.to_numeric(
    df[target],
    errors="raise"
)

groups = df[group_column]

# PREPROCESSING
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            "passthrough",
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)
# RANDOM FOREST MODEL
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),

        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)
# GROUP-BASED EVALUATION
number_of_molecules = groups.nunique()

if number_of_molecules < 2:
    raise ValueError(
        "At least two different molecules are required."
    )

cv = GroupKFold(
    n_splits=min(5, number_of_molecules)
)

predictions = cross_val_predict(
    model,
    X,
    y,
    groups=groups,
    cv=cv
)

mae = mean_absolute_error(y, predictions)

rmse = mean_squared_error(
    y,
    predictions
) ** 0.5

print("\nMODEL EVALUATION")
print("---------------------------")
print("MAE:", round(mae, 3), "ppm")
print("RMSE:", round(rmse, 3), "ppm")

print(
    "\nEvaluation uses held-out molecules, "
    "not randomly selected carbon atoms."
)
# TRAIN FINAL MODEL
model.fit(X, y)

joblib.dump(model, MODEL_FILE)

print("\nFinal model trained successfully!")
print("Model saved at:", MODEL_FILE)
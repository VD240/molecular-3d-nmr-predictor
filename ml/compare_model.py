import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_FILE = PROJECT_DIR / "dataset" / "carbon_nmr_features.csv"

df = pd.read_csv(DATASET_FILE)

numeric_features = [
    "atomic_number", "degree", "hydrogen_count",
    "is_aromatic", "is_in_ring", "formal_charge",
    "oxygen_count", "nitrogen_count", "bond_order_sum"
]

categorical_features = ["hybridization"]

X = df[numeric_features + categorical_features]
y = df["chemical_shift_ppm"]
groups = df["molecule_id"]

preprocessor = ColumnTransformer([
    ("numeric", "passthrough", numeric_features),
    ("categorical", OneHotEncoder(handle_unknown="ignore"),
     categorical_features)
])

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=3, random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42
    )
}

cv = GroupKFold(n_splits=min(5, groups.nunique()))

results = []

for name, algorithm in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", algorithm)
    ])

    predictions = cross_val_predict(
        pipeline, X, y,
        groups=groups,
        cv=cv
    )

    mae = mean_absolute_error(y, predictions)
    rmse = mean_squared_error(y, predictions) ** 0.5

    results.append({
        "Model": name,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3)
    })

result_df = pd.DataFrame(results)

print("\nREGRESSION MODEL COMPARISON")
print(result_df.to_string(index=False))

result_df.to_csv(
    PROJECT_DIR / "ml" / "model_comparison.csv",
    index=False
)


import pandas as pd
from pathlib import Path
import joblib

from rdkit import Chem

# PROJECT PATHS

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = PROJECT_DIR / "ml" / "carbon_nmr_model.joblib"
# LOAD TRAINED MODEL
if not MODEL_FILE.exists():
    raise FileNotFoundError(
        "Trained model not found. Run train_model.py first."
    )

model = joblib.load(MODEL_FILE)

# FEATURE EXTRACTION

def extract_features(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Invalid SMILES")

    features = []

    for atom in mol.GetAtoms():

        if atom.GetSymbol() != "C":
            continue

        oxygen_count = 0
        nitrogen_count = 0

        for neighbor in atom.GetNeighbors():

            if neighbor.GetSymbol() == "O":
                oxygen_count += 1

            elif neighbor.GetSymbol() == "N":
                nitrogen_count += 1

        bond_order_sum = sum(
            bond.GetBondTypeAsDouble()
            for bond in atom.GetBonds()
        )

        features.append({

            "atomic_number": atom.GetAtomicNum(),

            "degree": atom.GetDegree(),

            "hydrogen_count": atom.GetTotalNumHs(),

            "hybridization": str(atom.GetHybridization()),

            "is_aromatic": int(atom.GetIsAromatic()),

            "is_in_ring": int(atom.IsInRing()),

            "formal_charge": atom.GetFormalCharge(),

            "oxygen_count": oxygen_count,

            "nitrogen_count": nitrogen_count,

            "bond_order_sum": bond_order_sum

        })

    return mol, features

# PREDICT CHEMICAL SHIFTS

def predict_nmr(smiles):

    mol, features = extract_features(smiles)

    if not features:
        raise ValueError("No carbon atoms found.")

    feature_df=pd.DataFrame(features)
    predictions=model.predict(feature_df)

    results = []

    carbon_position = 0

    for atom in mol.GetAtoms():

        if atom.GetSymbol() != "C":
            continue

        results.append({

            "atom_index": atom.GetIdx(),

            "element": "C",

            "predicted_shift_ppm": round(
                float(predictions[carbon_position]), 2
            )

        })

        carbon_position += 1

    return results

# TEST PREDICTION

if __name__ == "__main__":

    test_smiles = "CCO"

    results = predict_nmr(test_smiles)

    print("\n13C NMR PREDICTION")
    print("-------------------------")

    for result in results:

        print(
            f"Carbon atom {result['atom_index']}: "
            f"{result['predicted_shift_ppm']} ppm"
        )
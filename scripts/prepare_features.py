import csv
from pathlib import Path
from rdkit import Chem

# Project paths
project_folder = Path(__file__).resolve().parent.parent

input_file = project_folder / "dataset" / "carbon_nmr_dataset.csv"
output_file = project_folder / "dataset" / "carbon_nmr_features.csv"

# Read original dataset
with open(input_file, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

features = []

for row in rows:

    mol = Chem.MolFromSmiles(row["smiles"])

    if mol is None:
        print("Invalid SMILES:", row["smiles"])
        continue

    atom_index = int(row["atom_index"])

    if atom_index >= mol.GetNumAtoms():
        print("Invalid atom index:", atom_index)
        continue

    atom = mol.GetAtomWithIdx(atom_index)

    # Count neighboring elements
    oxygen_count = 0
    nitrogen_count = 0

    for neighbor in atom.GetNeighbors():

        if neighbor.GetSymbol() == "O":
            oxygen_count += 1

        elif neighbor.GetSymbol() == "N":
            nitrogen_count += 1

    # Calculate bond order sum
    bond_order_sum = sum(
        bond.GetBondTypeAsDouble()
        for bond in atom.GetBonds()
    )

    features.append({
        "molecule_id": row["molecule_id"],
        "compound_name": row["compound_name"],
        "smiles": row["smiles"],
        "atom_index": atom_index,
        "atomic_number": atom.GetAtomicNum(),
        "degree": atom.GetDegree(),
        "hydrogen_count": atom.GetTotalNumHs(),
        "hybridization": str(atom.GetHybridization()),
        "is_aromatic": int(atom.GetIsAromatic()),
        "is_in_ring": int(atom.IsInRing()),
        "formal_charge": atom.GetFormalCharge(),
        "oxygen_count": oxygen_count,
        "nitrogen_count": nitrogen_count,
        "bond_order_sum": bond_order_sum,
        "chemical_shift_ppm": float(row["chemical_shift_ppm"]),
        "spectrum_type": row["spectrum_type"]
    })

# Save extracted features
if features:
    with open(output_file, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=features[0].keys()
        )

        writer.writeheader()
        writer.writerows(features)

print("Feature extraction completed!")
print("Total carbon atoms:", len(features))
print("Saved to:", output_file)
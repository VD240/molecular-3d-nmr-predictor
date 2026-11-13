import csv
from pathlib import Path
from rdkit import Chem

# Locate the project folder
project_folder = Path(__file__).resolve().parent.parent

# CSV output location
output_file = project_folder / "dataset" / "carbon_nmr_dataset.csv"

# Example molecules with illustrative carbon shifts
# These are synthetic values, NOT experimental measurements.
molecules = [
    {
        "name": "Ethanol",
        "smiles": "CCO",
        "shifts": [18.0, 58.0]
    },
    {
        "name": "Ethene",
        "smiles": "C=C",
        "shifts": [123.0, 123.0]
    },
    {
        "name": "Ethyne",
        "smiles": "C#C",
        "shifts": [72.0, 72.0]
    },
    {
        "name": "Acetic acid",
        "smiles": "CC(=O)O",
        "shifts": [20.0, 180.0]
    },
    {
        "name": "Benzene",
        "smiles": "c1ccccc1",
        "shifts": [128.0] * 6
    }
]

# Ensure dataset folder exists
output_file.parent.mkdir(parents=True, exist_ok=True)

columns = [
    "molecule_id",
    "compound_name",
    "smiles",
    "atom_index",
    "element",
    "hybridization",
    "is_aromatic",
    "chemical_shift_ppm",
    "solvent",
    "spectrometer_frequency_mhz",
    "doi",
    "spectrum_type"
]

with open(output_file, "w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()

    for molecule_id, item in enumerate(molecules, start=1):

        mol = Chem.MolFromSmiles(item["smiles"])

        if mol is None:
            raise ValueError(f"Invalid SMILES: {item['smiles']}")

        carbon_atoms = [
            atom for atom in mol.GetAtoms()
            if atom.GetSymbol() == "C"
        ]

        if len(carbon_atoms) != len(item["shifts"]):
            raise ValueError(
                f"Carbon count mismatch: {item['name']}"
            )

        for atom, shift in zip(carbon_atoms, item["shifts"]):

            writer.writerow({
                "molecule_id": f"MOL{molecule_id:03d}",
                "compound_name": item["name"],
                "smiles": item["smiles"],
                "atom_index": atom.GetIdx(),
                "element": atom.GetSymbol(),
                "hybridization": str(atom.GetHybridization()),
                "is_aromatic": atom.GetIsAromatic(),
                "chemical_shift_ppm": shift,
                "solvent": "",
                "spectrometer_frequency_mhz": "",
                "doi": "",
                "spectrum_type": "synthetic_example"
            })

print("Dataset generated successfully!")
print(f"Output file: {output_file}")
print(f"Total molecules: {len(molecules)}")
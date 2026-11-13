import csv
import joblib
import pandas as pd
from pathlib import Path
from collections import defaultdict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import AllChem,Descriptors,rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware


# Create FastAPI application
app = FastAPI(
    title="3D Molecular Visualizer API",
    description="Generate 3D molecular structures using RDKit",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
MODEL_FILE=(
    Path(__file__).resolve().parent
    /"ml"
    /"carbon_nmr_model.joblib"
)
model=joblib.load(MODEL_FILE)

# Define input format
class MoleculeRequest(BaseModel):
    smiles: str

def detect_functional_groups(mol):

    results = []

    for name, pattern in FUNCTIONAL_GROUPS.items():

        query = Chem.MolFromSmarts(pattern)

        matches = mol.GetSubstructMatches(query)

        if matches:

            results.append({
                "name": name,
                "atom_indices": [
                    list(match) for match in matches
                ],
                "count": len(matches)
            })

    return results
FUNCTIONAL_GROUPS = {
    "alcohol": "[CX4][OX2H]",

    "ether": "[OD2]([#6])[#6]",

    "aldehyde": "[CX3H1](=O)[#6]",

    "ketone": "[#6][CX3](=O)[#6]",

    "carboxylic_acid": "[CX3](=O)[OX2H1]",

    "ester": "[CX3](=O)[OX2H0][#6]",

    "amine": "[NX3;H2,H1;!$(NC=O)]",

    "amide": "[NX3][CX3](=[OX1])",

    "nitrile": "[C]#[N]",

    "alkene": "[C]=[C]",

    "alkyne": "[C]#[C]"
}
# Create API endpoint

@app.post("/generate-3d")
def generate_3d(request: MoleculeRequest):

    smiles = request.smiles

    # 1. Create molecule
   
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid SMILES"
        )

    # Keep a copy without explicit hydrogens
    heavy_mol = Chem.Mol(mol)

    # 2. Add explicit hydrogens

    mol = Chem.AddHs(mol)

    # 3. Generate 3D coordinates
 
    result = AllChem.EmbedMolecule(
        mol,
        randomSeed=42
    )

    if result != 0:
        raise HTTPException(
            status_code=422,
            detail="3D structure generation failed"
        )

    AllChem.MMFFOptimizeMolecule(mol)

    conf = mol.GetConformer()

   # 4. Extract all atom information
   
    atoms = []

    for atom in mol.GetAtoms():

        pos = conf.GetAtomPosition(atom.GetIdx())

        atoms.append({

            "index": atom.GetIdx(),

            "element": atom.GetSymbol(),

            "atomic_number": atom.GetAtomicNum(),

            "formal_charge": atom.GetFormalCharge(),

            "hybridization": str(
                atom.GetHybridization()
            ),

            "is_aromatic": atom.GetIsAromatic(),

            "total_hydrogens": atom.GetTotalNumHs(),

            "neighbors": [
                neighbor.GetIdx()
                for neighbor in atom.GetNeighbors()
            ],

            "x": round(pos.x, 3),

            "y": round(pos.y, 3),

            "z": round(pos.z, 3)
        })

    # 5. Extract bond information
    
    bonds = []

    for bond in mol.GetBonds():

        bonds.append({

            "from": bond.GetBeginAtomIdx(),

            "to": bond.GetEndAtomIdx(),

            "type": str(bond.GetBondType()),

            "is_aromatic": bond.GetIsAromatic()
        })

    # 6. Extract hydrogen information
    
    hydrogen_details = []

    for atom in mol.GetAtoms():

        if atom.GetSymbol() == "H":

            neighbors = atom.GetNeighbors()

            hydrogen_details.append({

                "hydrogen_index": atom.GetIdx(),

                "attached_to": neighbors[0].GetIdx(),

                "attached_element": neighbors[0].GetSymbol()

            })
            
 # 7. Extract carbon information
    
    carbon_details = []

    for atom in mol.GetAtoms():

        if atom.GetSymbol() == "C":

            hydrogen_count = sum(

                1 for neighbor in atom.GetNeighbors()

                if neighbor.GetSymbol() == "H"

            )

            carbon_details.append({

                "carbon_index": atom.GetIdx(),

                "hydrogen_count": hydrogen_count,

                "hybridization": str(
                    atom.GetHybridization()
                ),

                "is_aromatic": atom.GetIsAromatic(),

                "formal_charge": atom.GetFormalCharge(),

                "neighbors": [
                    neighbor.GetIdx()
                    for neighbor in atom.GetNeighbors()
                ]

            })

    # 8. Identify preliminary hydrogen groups
    
    hydrogen_groups = []

    for carbon in carbon_details:

        count = carbon["hydrogen_count"]

        hydrogen_groups.append({

            "carbon_index": carbon["carbon_index"],

            "group": f"CH{count}",

            "hydrogen_count": count

        })

    # 9. Identify hydrogen symmetry classes

    symmetry_ranks = list(
        Chem.CanonicalRankAtoms(
            mol,
            breakTies=False,
            includeChirality=True
        )
    )

    equivalent_hydrogens = {}

    for atom in mol.GetAtoms():

        if atom.GetSymbol() == "H":

            parent = atom.GetNeighbors()[0]

            symmetry_class = symmetry_ranks[atom.GetIdx()]

            key = (
                parent.GetIdx(),
                symmetry_class
            )

            equivalent_hydrogens.setdefault(
                str(key),
                []
            ).append(atom.GetIdx())

    hydrogen_equivalence = []

    for key, indices in equivalent_hydrogens.items():

        hydrogen_equivalence.append({

            "group_key": key,

            "hydrogen_indices": indices,

            "count": len(indices)

        })
    # 10. Detect functional groups
    
    functional_groups = detect_functional_groups(
        heavy_mol
    )

    # 11. Calculate molecular descriptors

    descriptors = {

        "molecular_formula":
            rdMolDescriptors.CalcMolFormula(heavy_mol),

        "molecular_weight":
            round(Descriptors.MolWt(heavy_mol), 3),

        "logP":
            round(Descriptors.MolLogP(heavy_mol), 3),

        "TPSA":
            round(Descriptors.TPSA(heavy_mol), 3),

        "hydrogen_bond_donors":
            Descriptors.NumHDonors(heavy_mol),

        "hydrogen_bond_acceptors":
            Descriptors.NumHAcceptors(heavy_mol),

        "rotatable_bonds":
            Descriptors.NumRotatableBonds(heavy_mol),

        "number_of_rings":
            Descriptors.RingCount(heavy_mol)

    }

    # 12. Return complete information

    return {

        "smiles": smiles,

        "total_atoms": mol.GetNumAtoms(),

        "total_bonds": mol.GetNumBonds(),

        "atoms": atoms,

        "bonds": bonds,

        "hydrogens": hydrogen_details,

        "carbon_details": carbon_details,

        "hydrogen_groups": hydrogen_groups,

        "hydrogen_equivalence": hydrogen_equivalence,

        "functional_groups": functional_groups,

        "descriptors": descriptors

    }

@app.post("/generate-2d")
def generate_2d(request: MoleculeRequest):

    # Read SMILES
    smiles = request.smiles

    # Convert SMILES into molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid SMILES"
            )


    atom_details = []

    for atom in mol.GetAtoms():
        atom_details.append({
        "index": atom.GetIdx(),
        "element": atom.GetSymbol(),
        "atomic_number": atom.GetAtomicNum(),
        "total_hydrogens":atom.GetTotalNumHs(),
        "neighbors": [
            neighbor.GetIdx()
            for neighbor in atom.GetNeighbors()
        ]
    })
  

    # Generate 2D coordinates
    AllChem.Compute2DCoords(mol)

    # Create drawing object
    drawer = rdMolDraw2D.MolDraw2DSVG(500, 400)

    options=drawer.drawOptions()
    palette={
        6: (0.25,0.25,0.25),#Carbon:darkgray
        8: (0.90,0.10,0.10),#Oxygen:red
        7: (0.10,0.25,0.90),#Nitrogen:blue
        16:(0.95,0.75,0.10) #Sulfur:yellow
    }
    drawer.drawOptions().setAtomPalette(palette)
    options.useBWAtomPalette=False
    options.addAtomIndices=True
                               
     # Draw molecule
    drawer.DrawMolecule(mol)

    # Finish drawing
    drawer.FinishDrawing()

    # Get SVG image
    svg = drawer.GetDrawingText()

    return {
          "svg": svg,
          "atoms": atom_details
    }
@app.get("/nmr-spectrum/{molecule_name}")
def get_nmr_spectrum(molecule_name: str):

    csv_file = (
        Path(__file__).resolve().parent
        / "dataset"
        / "carbon_nmr_features.csv"
    )

    if not csv_file.exists():
        raise HTTPException(
            status_code=404,
            detail="NMR dataset not found"
        )

    with open(csv_file, "r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
        print("CSV HEADERS:",rows[0].keys() 
              if rows else "CSV IS EMPTY")

    molecule_rows = [
        row for row in rows
        if row["compound_name"].lower() == molecule_name.lower()
    ]

    if not molecule_rows:
        raise HTTPException(
            status_code=404,
            detail="Molecule not found"
        )

    peaks = defaultdict(list)

    for row in molecule_rows:

        shift = float(row["chemical_shift_ppm"])

        peaks[shift].append({
            "atom_index": int(row["atom_index"]),
            "element": "C"
        })

    spectrum = []

    for shift, atoms in sorted(peaks.items()):

        spectrum.append({
            "chemical_shift_ppm": shift,
            "intensity": len(atoms),
            "atoms": atoms
        })

    return {
        "molecule": molecule_name,
        "total_peaks": len(spectrum),
        "spectrum": spectrum
    }
@app.post("/nmr-spectrum-by-smiles")
def get_nmr_spectrum_by_smiles(request: MoleculeRequest):

    csv_file = (
        Path(__file__).resolve().parent
        / "dataset"
        / "carbon_nmr_features.csv"
    )

    if not csv_file.exists():
        raise HTTPException(
            status_code=404,
            detail="NMR dataset not found"
        )

    with open(csv_file, "r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    molecule_rows = [
        row for row in rows
        if row["smiles"].strip() == request.smiles.strip()
    ]

    if not molecule_rows:
        raise HTTPException(
            status_code=404,
            detail="No NMR data found for this SMILES"
        )

    peaks = defaultdict(list)

    for row in molecule_rows:

        shift = float(row["chemical_shift_ppm"])

        peaks[shift].append({
            "atom_index": int(row["atom_index"]),
            "element": "C"
        })

    spectrum = []

    for shift, atoms in sorted(peaks.items()):

        spectrum.append({
            "chemical_shift_ppm": shift,
            "intensity": len(atoms),
            "atoms": atoms
        })

    return {
        "molecular_name":molecule_rows[0]["compound_name"],
        "smiles": request.smiles,
        "total_peaks": len(spectrum),
        "spectrum": spectrum
    }
@app.post("/predict-nmr")
def predict_nmr_endpoint(request: MoleculeRequest):

    smiles = request.smiles

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid SMILES"
        )

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

    if not features:
        raise HTTPException(
            status_code=400,
            detail="No carbon atoms found"
        )

    feature_df = pd.DataFrame(features)

    predictions = model.predict(feature_df)

    results = []

    carbon_position = 0

    for atom in mol.GetAtoms():

        if atom.GetSymbol() == "C":

            results.append({
                "atom_index": atom.GetIdx(),
                "element": "C",
                "predicted_shift_ppm": round(
                    float(predictions[carbon_position]), 2
                )
            })

            carbon_position += 1

    return {
        "smiles": smiles,
        "total_carbon_atoms": len(results),
        "predictions": results
    }
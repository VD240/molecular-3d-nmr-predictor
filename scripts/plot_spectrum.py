import csv
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

project_folder = Path(__file__).resolve().parent.parent

input_file = project_folder / "dataset" / "carbon_nmr_features.csv"
output_file = project_folder / "dataset" / "all_spectra_with_atoms.png"

# Read dataset
with open(input_file, "r", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

molecules = sorted(set(row["compound_name"] for row in rows))

fig, axes = plt.subplots(
    len(molecules),
    1,
    figsize=(12, 3 * len(molecules))
)

# Ensure axes is iterable
if len(molecules) == 1:
    axes = [axes]

for ax, molecule_name in zip(axes, molecules):

    molecule_rows = [
        row for row in rows
        if row["compound_name"] == molecule_name
    ]

    # Group carbon atoms by chemical shift
    peak_groups = defaultdict(list)

    for row in molecule_rows:
        shift = float(row["chemical_shift_ppm"])
        atom_index = row["atom_index"]

        peak_groups[shift].append(atom_index)

    # Draw peaks and atom labels
    for shift, atom_indices in peak_groups.items():

        intensity = len(atom_indices)

        ax.vlines(
            shift,
            0,
            intensity,
            linewidth=2
        )

        label = ", ".join(atom_indices)

        ax.annotate(
            f"Atom {label}",
            xy=(shift, intensity),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8
        )

    ax.set_title(molecule_name)
    ax.set_xlim(220, 0)
    ax.set_ylim(0, len(molecule_rows) + 2)
    ax.set_ylabel("Intensity")
    ax.grid(alpha=0.2)

axes[-1].set_xlabel("Chemical Shift (ppm)")

fig.suptitle("13C NMR Spectrum with Carbon Atom Mapping", fontsize=15)

plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()

print("Spectrum with atom mapping generated!")
print("Saved at:", output_file)
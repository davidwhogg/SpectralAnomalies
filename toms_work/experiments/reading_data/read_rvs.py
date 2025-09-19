# read_rvs.py
# The one where Tom learns what stellar spectra look like

from pathlib import Path

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("mpl_drip.custom")

# Get the files, check existence
DATA_LOC = Path("../../data")
SPECTRA = DATA_LOC / "dr3-rvs-all.hdf5"
META = DATA_LOC / "dr3-source-meta.csv"
assert SPECTRA.is_file()
assert META.is_file()

# Read the hdf5 file
f_spec = h5.File(SPECTRA, "r")

# Grid in wavelengths and interesting source ID
λ_GRID = np.linspace(846, 870, 2401)  # nm
SOURCE_ID = 4264967400467007232

# Get the spectrum
idx = np.where(f_spec["source_id"][:] == SOURCE_ID)[0][0]
spec = f_spec["flux"][idx]

# Nice plot
fig, ax = plt.subplots(figsize=[9, 3], dpi=200, layout="compressed")
ax.plot(λ_GRID, spec, lw=2)
ax.set_xlabel(r"$\lambda$ [nm]")
ax.set_ylabel("Normalised Flux")
ax.set_title(f"Gaia DR3 {SOURCE_ID}")
plt.show()

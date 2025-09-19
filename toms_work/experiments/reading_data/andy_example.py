import h5py as h5
import numpy as np
import matplotlib.pyplot as plt

fp = h5.File("/Users/andycasey/Downloads/dr3-rvs-all.hdf5", "r")


wavelength = np.linspace(846, 870, 2401) # nm

source_id = 4264967400467007232

index = np.where(fp["source_id"][:] == source_id)[0][0]

fig, ax = plt.subplots(figsize=(14, 3))

ax.plot(wavelength, fp["flux"][index], c='k')
ax.set_title(f"Gaia DR3 {source_id}")
ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel("Normalized flux [-]")
fig.tight_layout()
fig.savefig("example.png", dpi=300)

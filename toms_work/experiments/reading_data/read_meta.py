# read_meta.py
# Check I still understand CSV files and make some cute plots

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

plt.style.use("mpl_drip.custom")

# Get the files, check existence
DATA_LOC = Path("../../data")
SPECTRA = DATA_LOC / "dr3-rvs-all.hdf5"
META = DATA_LOC / "dr3-source-meta.csv"
assert SPECTRA.is_file()
assert META.is_file()

SOURCE_ID = 4264967400467007232

# Read the csv
lf_meta = pl.scan_csv(META)
# _ = [print(x) for x in lf_meta.columns]
lf_meta_selected = lf_meta.select(
    [
        "source_id",
        "parallax",
        "bp_rp",
        "phot_g_mean_mag",
    ]
)
df_met = lf_meta_selected.collect()

source_id = df_met["source_id"].to_numpy()
bp_rp = df_met["bp_rp"].to_numpy()
phot_g_mean_mag = df_met["phot_g_mean_mag"].to_numpy()
parallax = df_met["parallax"].to_numpy()


def nans_mask(arrs):
    nans = np.isnan(np.array(arrs))
    return np.logical_not(np.any(nans, axis=0))


any_nans = nans_mask([bp_rp, phot_g_mean_mag, parallax])
source_id = source_id[any_nans]
bp_rp = bp_rp[any_nans]
phot_g_mean_mag = phot_g_mean_mag[any_nans]
parallax = parallax[any_nans]

neg_parallaxes = parallax < 0
source_id = source_id[~neg_parallaxes]
bp_rp = bp_rp[~neg_parallaxes]
phot_g_mean_mag = phot_g_mean_mag[~neg_parallaxes]
parallax = parallax[~neg_parallaxes]

abs_mag_G = phot_g_mean_mag + 5 * np.log10(parallax / 1000) + 5

source_idx = np.where(source_id == SOURCE_ID)[0][0]

fig, ax = plt.subplots(dpi=200, figsize=[9, 9], layout="compressed")
ax.scatter(bp_rp, abs_mag_G, s=0.5, alpha=0.03, c="C0")
ax.scatter(
    bp_rp[source_idx],
    abs_mag_G[source_idx],
    s=60,
    alpha=1,
    c="C3",
    marker="x",
    zorder=10,
)
ax.set_ylabel("Absolute Mag G band")
ax.set_xlabel("BP - RP")
ax.set_ylim(13, -6)
ax.set_xlim(-0.2, 5.2)
plt.show()

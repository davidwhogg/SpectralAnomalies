# collect.py
# Get the stuff and things

from pathlib import Path

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# Get the files, check existence
DATA_LOC = Path("../../data")
SPECTRA = DATA_LOC / "dr3-rvs-all.hdf5"
META = DATA_LOC / "dr3-source-meta.csv"
assert SPECTRA.is_file()
assert META.is_file()

TARGET_ID = 4264967400467007232

rng = np.random.default_rng(0)
plt.style.use("mpl_drip.custom")


def nans_mask(arrs):
    nans = np.isnan(np.array(arrs))
    return np.logical_not(np.any(nans, axis=0))


def read_meta(filter_nans=True, filter_neg_parallax=True):
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

    if filter_nans:
        any_nans = nans_mask([bp_rp, phot_g_mean_mag, parallax])
        source_id = source_id[any_nans]
        bp_rp = bp_rp[any_nans]
        phot_g_mean_mag = phot_g_mean_mag[any_nans]
        parallax = parallax[any_nans]

    if filter_neg_parallax:
        neg_parallaxes = parallax < 0
        source_id = source_id[~neg_parallaxes]
        bp_rp = bp_rp[~neg_parallaxes]
        phot_g_mean_mag = phot_g_mean_mag[~neg_parallaxes]
        parallax = parallax[~neg_parallaxes]

    return source_id, bp_rp, phot_g_mean_mag, parallax


def find_source_index(source_id, target_id=TARGET_ID):
    return np.where(source_id == target_id)[0][0]


def read_spectra():
    f_spec = h5.File(SPECTRA, "r")
    λ_grid = np.linspace(846, 870, 2401)  # nm
    return (
        λ_grid,
        f_spec["source_id"][:],
        f_spec["flux"],
        f_spec["flux_error"],
    )


def compute_abs_mag(phot_g_mean_mag, parallax):
    abs_mag_G = phot_g_mean_mag + 5 * np.log10(parallax / 1000) + 5
    return abs_mag_G


def get_data(thresh_bp_rp=0.05, thresh_abs_mag=0.05, clip_edge_pix=20, mask="ellipse"):
    # Read meta and spectra
    source_id, bp_rp, phot_g_mean_mag, parallax = read_meta()
    spec_λ, spec_source_id, spec_flux, spec_u_flux = read_spectra()
    abs_mag_G = compute_abs_mag(phot_g_mean_mag, parallax)
    target_idx = find_source_index(source_id, TARGET_ID)

    # Find things that are similar to the target in colour-magnitude space, in an ellipse
    target_bp_rp = bp_rp[target_idx]
    target_abs_mag_G = abs_mag_G[target_idx]
    bp_rp_diff = np.abs(bp_rp - target_bp_rp)
    abs_mag_diff = np.abs(abs_mag_G - target_abs_mag_G)
    if mask == "ellipse":
        similar_mask = (bp_rp_diff / thresh_bp_rp) ** 2 + (
            abs_mag_diff / thresh_abs_mag
        ) ** 2 < 1
    elif mask == "rectangle":
        similar_mask = (bp_rp_diff < thresh_bp_rp) & (abs_mag_diff < thresh_abs_mag)
    print(f"Found {np.sum(similar_mask)} similar sources")

    # Get the spectra of the similar sources
    similar_spec_idxs = np.sort(
        [find_source_index(spec_source_id, sid) for sid in source_id[similar_mask]]
    )
    similar_specs = spec_flux[similar_spec_idxs]

    # Data to fit
    Y = similar_specs.copy()
    W = 1.0 / spec_u_flux[similar_spec_idxs] ** 2

    spec_nans_mask = nans_mask([Y, W])
    Y[~spec_nans_mask] = np.nan
    W[~spec_nans_mask] = np.nan
    Y = np.nan_to_num(Y)
    W = np.nan_to_num(W)

    l_ind, u_ind = clip_edge_pix, Y.shape[1] - clip_edge_pix
    Y = Y[:, l_ind:u_ind]
    W = W[:, l_ind:u_ind]
    spec_λ = spec_λ[l_ind:u_ind]

    # Get also the bp_rp and abs_mag_G of the similar sources
    sim_bp_rp = bp_rp[similar_mask]
    sim_abs_mag_G = abs_mag_G[similar_mask]

    return Y, W, spec_λ, sim_bp_rp, sim_abs_mag_G


THRESH_BP_RP = 0.2
THRESH_ABS_MAG = 0.2

# NOTE: Yes this is repeated code, but it's just to make some plots if you want to run this file directly
if __name__ == "__main__":
    # Read meta and spectra
    source_id, bp_rp, phot_g_mean_mag, parallax = read_meta()
    spec_λ, spec_source_id, spec_flux, spec_u_flux = read_spectra()
    abs_mag_G = compute_abs_mag(phot_g_mean_mag, parallax)
    target_idx = find_source_index(source_id, TARGET_ID)
    # target_idx = 56792
    target_spec_idx = find_source_index(spec_source_id, TARGET_ID)
    print(
        f"Sanity check source IDs: {source_id[target_idx]} {spec_source_id[target_spec_idx]}"
    )

    # Find things that are similar to the target in colour-magnitude space, in an ellipse
    target_bp_rp = bp_rp[target_idx]
    target_abs_mag_G = abs_mag_G[target_idx]
    bp_rp_diff = np.abs(bp_rp - target_bp_rp)
    abs_mag_diff = np.abs(abs_mag_G - target_abs_mag_G)
    ellipse_mask = (bp_rp_diff / THRESH_BP_RP) ** 2 + (
        abs_mag_diff / THRESH_ABS_MAG
    ) ** 2 < 1
    rectangle_mask = (bp_rp_diff < THRESH_BP_RP) & (abs_mag_diff < THRESH_ABS_MAG)
    similar_mask = ellipse_mask  # or rectangle_mask
    print(f"Found {np.sum(similar_mask)} similar sources")

    # Plot the CMD with the target and similar sources highlighted
    fig, ax = plt.subplots(dpi=100, figsize=[8, 4], layout="compressed")
    ax.scatter(bp_rp, abs_mag_G, s=0.5, alpha=0.05, c="C0", zorder=0)
    ax.scatter(
        bp_rp[target_idx],
        abs_mag_G[target_idx],
        s=50,
        alpha=1,
        c="C1",
        label="Target",
        zorder=2,
    )
    ax.scatter(
        bp_rp[similar_mask],
        abs_mag_G[similar_mask],
        s=5,
        alpha=0.05,
        c="C2",
        label="Similar",
        zorder=1,
    )
    ax.set_ylabel("Absolute Mag G band")
    ax.set_xlabel("BP - RP")
    ypad_fac = 11
    xpad_fac = 15
    # ypad_fac = 2
    # xpad_fac = 2
    y_ext = THRESH_ABS_MAG * ypad_fac
    x_ext = THRESH_BP_RP * xpad_fac
    ax.set_ylim(target_abs_mag_G + y_ext, target_abs_mag_G - y_ext)
    # ax.set_xlim(target_bp_rp - x_ext, target_bp_rp + x_ext)
    ax.set_xlim(-0.5, 3.5)
    ax.legend()
    plt.show()

    # Get the spectra of the similar sources
    similar_spec_idxs = np.sort(
        [find_source_index(spec_source_id, sid) for sid in source_id[similar_mask]]
    )
    similar_specs = spec_flux[similar_spec_idxs]
    similar_ids = spec_source_id[similar_spec_idxs]
    similar_bp_rp = bp_rp[similar_mask]
    similar_abs_mag_G = abs_mag_G[similar_mask]
    target_similar_idx = find_source_index(similar_ids, TARGET_ID)

    # Plot the target spectrum and some random similar spectra
    fig, ax = plt.subplots(figsize=[7, 4], dpi=100, layout="compressed")
    ax.plot(
        spec_λ,
        similar_specs[target_similar_idx],
        lw=1,
        alpha=1,
        c="C3",
        label="Target",
        zorder=10,
    )
    for i in rng.choice(similar_specs.shape[0], size=40, replace=False):
        ax.plot(spec_λ, similar_specs[i], lw=1, alpha=0.5, c="C2", label="Similar")
    ax.set_xlabel(r"$\lambda$ [nm]")
    ax.set_ylabel("Normalised Flux")
    # ax.set_xlim(849.5, 850.5)
    # ax.set_ylim(0.8, 1.00)
    ax.set_title(f"Gaia DR3 {TARGET_ID} and similar sources in CM-space")
    plt.show()

    # Data to fit
    Y = similar_specs.copy()
    W = 1.0 / spec_u_flux[similar_spec_idxs] ** 2

    # Plot the number of nan weights per pixel
    plt.figure(figsize=[8, 4], dpi=100, layout="compressed")
    plt.plot(np.sum(np.isnan(W), axis=0))
    plt.xlabel("Pixel index")
    plt.ylabel("Number of nan weights")
    plt.title("Number of nan weights per pixel")
    plt.show()

    # Plot the number of zeros in the data (Y) per pixel
    plt.figure(figsize=[8, 4], dpi=100, layout="compressed")
    plt.plot(np.sum(Y == 0, axis=0))
    plt.xlabel("Pixel index")
    plt.ylabel("Number of zero fluxes")
    plt.title("Number of zero fluxes per pixel")
    plt.show()

    spec_nans_mask = nans_mask([Y, W])
    Y[~spec_nans_mask] = np.nan
    W[~spec_nans_mask] = np.nan
    Y = np.nan_to_num(Y)
    W = np.nan_to_num(W)

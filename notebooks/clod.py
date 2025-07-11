import numpy as np
from astroquery.gaia import Gaia
from astropy.table import Table
import os

def find_rvs_sources_gspphot(teff_min=4810, teff_max=6200,
                            logg_min=1.0, logg_max=3.0,
                            grvs_mag_max=11.0, n_sources=500):
    query = f"""
    SELECT TOP {n_sources}
        source_id, ra, dec, phot_g_mean_mag, grvs_mag,
        radial_velocity, radial_velocity_error,
        teff_gspphot, logg_gspphot, mh_gspphot,
        bp_rp, parallax
    FROM gaiadr3.gaia_source
    WHERE has_rvs = 't'
    AND grvs_mag <= {grvs_mag_max}
    AND teff_gspphot BETWEEN {teff_min} AND {teff_max}
    AND logg_gspphot BETWEEN {logg_min} AND {logg_max}
    AND teff_gspphot IS NOT NULL
    AND logg_gspphot IS NOT NULL
    AND radial_velocity IS NOT NULL
    ORDER BY grvs_mag ASC
    """
    job = Gaia.launch_job_async(query)
    sources = job.get_results()
    print(f"\nFound {len(sources)} sources matching criteria")
    return sources

def download_rvs_spectrum(source_id, output_dir="rvs_data_cache"):
    """
    Download RVS spectrum for a single source.
    Returns wavelength and flux arrays.
    """
    # Check if spectrum already exists in cache
    cache_file = os.path.join(output_dir, f"rvs_{source_id}.npz")
    if os.path.exists(cache_file):
        data = np.load(cache_file)
        return data['wavelength'], data['flux'], data['flux_error']
    try:
        # Download spectrum
        retrieval_type = 'RVS'
        data_structure = 'INDIVIDUAL'
        data_release = 'Gaia DR3'

        datalink_products = Gaia.load_data(
            ids=[str(source_id)],
            data_release=data_release,
            retrieval_type=retrieval_type,
            data_structure=data_structure,
            verbose=False
        )

        if not datalink_products:
            return None, None

        product_key = f"RVS-Gaia DR3 {source_id}.xml"

        if product_key not in datalink_products:
            return None, None

        # Extract spectrum
        votable = datalink_products[product_key][0]
        spectrum_table = votable.to_table()

        wavelength = np.array(spectrum_table['wavelength'])  # in nm
        flux = np.array(spectrum_table['flux'])  # normalized
        flux_error = np.array(spectrum_table['flux_error'])

        # Save to cache
        os.makedirs(output_dir, exist_ok=True)
        np.savez(cache_file, wavelength=wavelength,
                 flux=flux, flux_error=flux_error)

        return wavelength, flux, flux_error

    except Exception as e:
        print(f"Error downloading spectrum for source {source_id}: {e}")
        return None, None

def download_multiple_spectra(sources, max_spectra=None):
    if max_spectra is None:
        max_spectra = len(sources)

    spectra_data = {}
    successful_downloads = 0

    print(f"\nDownloading RVS spectra for up to {max_spectra} sources...")

    # Check column names (Gaia returns uppercase)
    if 'SOURCE_ID' in sources.colnames:
        source_id_col = 'SOURCE_ID'
    else:
        source_id_col = 'source_id'

    for i, source in enumerate(sources[:max_spectra]):
        source_id = source[source_id_col]

        if i % 10 == 0:
            print(f"Progress: {i}/{max_spectra} spectra processed...")

        wavelength, flux, flux_error = download_rvs_spectrum(source_id)

        if wavelength is not None and flux is not None:
            spectra_data[source_id] = (wavelength, flux, flux_error)
            successful_downloads += 1

    print(f"\nSuccessfully downloaded {successful_downloads} spectra")

    return spectra_data

def create_spectral_matrices(spectra_data, wavelength_grid=None, fill_value=1.0):
    """
    Create a matrices Y, W where each row is a spectrum or its invvar weight

    Parameters:
    -----------
    spectra_data : dict
        Dictionary of (wavelength, flux) tuples
    wavelength_grid : array-like, optional
        Common wavelength grid. If None, uses the first spectrum's grid
    fill_value : float
        Value to use for replacing NaN/Inf (default: 1.0 for continuum)
    n_clip_lower : int
        Number of pixels to clip from the lower wavelength end
    n_clip_upper : int
        Number of pixels to clip from the upper wavelength end

    Returns:
    --------
    Y : np.ndarray
        Matrix of shape (n_spectra, n_wavelengths)
    wavelength_grid : np.ndarray
        Wavelength grid used
    source_ids : list
        List of source IDs in same order as rows of Y
    W : np.ndarray
        invvars for Y, with bad pixels zeroed out
    """

    source_ids = list(spectra_data.keys())
    n_spectra = len(source_ids)

    # Use first spectrum to define wavelength grid if not provided
    if wavelength_grid is None:
        wavelength_grid = spectra_data[source_ids[0]][0]
    n_wavelengths = len(wavelength_grid)

    # Initialize spectral matrix and bad pixel mask
    Y = np.zeros((n_spectra, n_wavelengths)) + np.nan
    W = np.zeros((n_spectra, n_wavelengths))

    # Track statistics
    total_bad_pixels = 0
    spectra_with_bad_pixels = 0

    # Fill matrix
    for i, source_id in enumerate(source_ids):
        wavelength, flux, flux_error = spectra_data[source_id]

        # Make weights / invvars
        invvar = 1. / flux_error ** 2
        bad_pixels = (np.isnan(flux) | np.isinf(flux)) | ((invvar < 1.) | np.isnan(invvar))
        if np.any(bad_pixels):
            spectra_with_bad_pixels += 1
            total_bad_pixels += np.sum(bad_pixels)

            # Replace bad pixels with fill_value
            flux = np.where(bad_pixels, fill_value, flux)
            invvar = np.where(bad_pixels, 0., invvar)

        Y[i, :] = flux
        W[i, :] = invvar

    print(f"\nBad pixel statistics:")
    print(f"  Spectra with bad pixels: {spectra_with_bad_pixels}/{n_spectra}")
    print(f"  Total bad pixels: {total_bad_pixels}")
    print(f"  Bad pixels replaced with: {fill_value}")
    print(f"  Bad weights (inverse variances) replaced with: {0}")

    return Y, wavelength_grid, np.array(source_ids), W


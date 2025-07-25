import os
import numpy as np
import pickle
import requests
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from urllib.parse import urljoin
import time
from pathlib import Path

class SDSSLRGProcessor:
    def __init__(self, cache_dir='./sdss_cache', dr='dr16'):
        """
        Initialize SDSS LRG processor with local caching
        
        Parameters:
        -----------
        cache_dir : str
            Directory to store cached spectra
        dr : str
            Data release (e.g., 'dr16', 'dr17')
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.dr = dr
        
        # Common rest-frame wavelength grid (Angstroms) MAGIC NUMBERS
        self.rest_wave_grid = 10. ** np.arange(np.log10(2900.), np.log10(7800.), 0.0001)

        # SDSS base URL for spectra
        self.base_url = f"https://data.sdss.org/sas/{dr}/eboss/spectro/redux/"
        
        # Cache files
        self.spectra_cache = self.cache_dir / 'spectra_cache.pkl'
        self.processed_cache = self.cache_dir / 'processed_spectra.pkl'
        
    def download_spall(self):
        """Download the spAll file containing all SDSS spectroscopic objects"""
        spall_file = self.cache_dir / 'spAll-v5_13_0.fits'
        
        if spall_file.exists():
            print("Loading cached spAll file...")
            return str(spall_file)
        
        print("Downloading spAll file (this may take a while - ~1GB file)...")
        spall_url = f"https://data.sdss.org/sas/{self.dr}/eboss/spectro/redux/v5_13_0/spAll-v5_13_0.fits"
        
        try:
            response = requests.get(spall_url, stream=True, timeout=300)
            response.raise_for_status()
            
            # Download with progress indication
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(spall_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if np.random.uniform() < 0.01 and total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\rDownloading spAll: {percent:.1f}%", end='', flush=True)
            
            print(f"\nspAll file downloaded successfully: {spall_file}")
            return str(spall_file)
            
        except requests.exceptions.RequestException as e:
            print(f"Failed to download spAll file: {e}")
            return None
    
    def get_lrg_sample(self, max_objects, z_min, z_max):
        """
        Get a sample of real LRG objects from SDSS spAll file.
        This queries the actual SDSS spectroscopic database.
        """
        # Download spAll file
        spall_file = self.download_spall()
        
        print("Loading spAll file and selecting LRGs...")
        try:
            # Load the spAll file
            with fits.open(spall_file) as hdul:
                data = hdul[1].data
            
            # Select LRGs based on SDSS criteria
            # CLASS = 'GALAXY' and typical LRG color/magnitude cuts
            lrg_mask = (
                (data['CLASS'] == 'GALAXY') &
                (data['Z'] > z_min) & (data['Z'] < z_max) &
                (data['Z_ERR'] > 0) & (data['Z_ERR'] < 0.001) &  # Very good redshift quality; too good?
                (data['ZWARNING'] == 0) # No redshift warnings
                & (data['SN_MEDIAN_ALL'] > 4.0)  # Decent S/N
                # Additional LRG-like criteria (adjust as needed)
                # & (data['MODELMAG'][:, 1] > 17.0) & (data['MODELMAG'][:, 1] < 19.2)  # r-band magnitude
                # & (data['MODELMAG'][:, 2] - data['MODELMAG'][:, 3] > 0.5)  # Red color cut
            )
            
            # Get LRG indices and shorten the data immediately
            lrg_indices = np.where(lrg_mask)[0]
            if len(lrg_indices) == 0:
                print("No LRGs found with current criteria, using fallback")
                return self._get_fallback_sample(max_objects, z_min, z_max)
            data = data[lrg_indices]
            
            # Make identifier strings
            objids = np.array([f"{d['PLATE']:05d}-{d['FIBERID']:04d}-{d['MJD']:05d}" for d in data])

            # Order by a silly but reproducible fact
            dijbos = np.array([o[::-1] for o in objids])
            I = np.argsort(dijbos)
            data = data[I]
            objids = objids[I]
            
            # Subsample if we have too many
            if len(data) > max_objects:
                data = data[:max_objects]
                objids = objids[:max_objects]
            
            # Extract the data
            sample = {
                'plate': data['PLATE'].tolist(),
                'mjd': data['MJD'].tolist(),
                'fiberid': data['FIBERID'].tolist(),
                'z': data['Z'].tolist(),
                'objid': objids.tolist()
            }
            
            print(f"Selected {len(data)} LRGs from spAll file")
            print(f"Empirical redshift range: {np.min(sample['z']):.3f} - {np.max(sample['z']):.3f}")
            
            return sample
            
        except Exception as e:
            print(f"Error processing spAll file: {e}")
            assert False
    
    def get_spectrum_url(self, plate, mjd, fiberid):
        """Construct URL for SDSS spectrum file"""
        run2d = "v5_13_0"  # Common run2d version for DR16
        filename = f"spec-{plate:04d}-{mjd}-{fiberid:04d}.fits"
        url = f"{self.base_url}{run2d}/spectra/full/{plate:04d}/{filename}"
        return url
    
    def download_spectrum(self, plate, mjd, fiberid, objid):
        """Download and cache a single spectrum"""
        cache_file = self.cache_dir / f"{plate:04d}/spec_{plate:04d}_{mjd}_{fiberid:04d}.fits"
        
        # Check if already cached
        if cache_file.exists():
            return str(cache_file)
        else:
            os.makedirs(self.cache_dir / f"{plate:04d}", exist_ok=True)
        
        # Download spectrum
        url = self.get_spectrum_url(plate, mjd, fiberid)
        print(f"Downloading spectrum for {objid}: {url}")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save to cache
            with open(cache_file, 'wb') as f:
                f.write(response.content)
            
            time.sleep(0.1)  # Be nice to SDSS servers
            return str(cache_file)
            
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {objid}: {e}")
            return None
    
    def load_spectrum(self, file_path):
        """Load spectrum from FITS file"""
        try:
            with fits.open(file_path) as hdul:
                # SDSS spectrum structure
                flux = hdul[1].data['flux']
                loglam = hdul[1].data['loglam']
                ivar = hdul[1].data['ivar']
                
                # Convert log-lambda to linear wavelength
                wave = 10**loglam
                
                return {
                    'wavelength': wave,
                    'flux': flux,
                    'ivar': ivar
                }
        except Exception as e:
            print(f"Error loading spectrum {file_path}: {e}")
            return None
    
    def process_spectrum(self, spectrum_data, redshift):
        """Process spectrum to rest-frame and common wavelength grid"""
        wave = spectrum_data['wavelength']
        flux = spectrum_data['flux']
        ivar = spectrum_data['ivar']
        
        # Convert to rest-frame wavelength
        rest_wave = wave / (1 + redshift)
        
        # Apply flux correction for redshift
        rest_flux = flux * (1 + redshift)
        rest_ivar = ivar / (1 + redshift) ** 2
        
        # Create mask for valid data
        foo = np.nanmedian(flux)
        tiny = 0.01 / foo ** 2
        valid_mask = (
            np.isfinite(rest_flux) & 
            (rest_ivar > tiny) & 
            (rest_wave >= self.rest_wave_grid.min()) & 
            (rest_wave <= self.rest_wave_grid.max())
        )
        
        if np.sum(valid_mask) < 2000:
            print("Not enough valid data points for interpolation")
            return None
        
        # Interpolate to common wavelength grid
        try:
            # Use nearest interpolation -- this is INSANE
            f_interp = interp1d(
                rest_wave[valid_mask],
                rest_flux[valid_mask],
                kind='nearest',
                bounds_error=False,
                fill_value=foo
            )
            
            # Use nearest-neighbor interpolation for inverse variance (more appropriate for weights)
            ivar_interp = interp1d(
                rest_wave[valid_mask],
                spectrum_data['ivar'][valid_mask],
                kind='nearest',
                bounds_error=False,
                fill_value=tiny  # Use something nonzero for invalid regions (very large error)
            )

            # Interpolate to common grid
            interp_flux = f_interp(self.rest_wave_grid)
            interp_ivar = ivar_interp(self.rest_wave_grid)
            
            # Do one more conservative thing with the ivars.
            interp_ivar = np.minimum(interp_ivar, np.roll(interp_ivar, 1))
            interp_ivar = np.minimum(interp_ivar, np.roll(interp_ivar, -1))

            return {
                'wavelength': self.rest_wave_grid.copy(),
                'flux': interp_flux,
                'ivar': interp_ivar,
                'redshift': redshift
            }
            
        except Exception as e:
            print(f"Error interpolating spectrum: {e}")
            return None
    
    def process_lrg_sample(self, max_objects=100, z_min=0.2, z_max=0.55, force_reprocess=False):
        """Process entire LRG sample with caching"""
        
        # Check if processed data exists
        if self.processed_cache.exists() and not force_reprocess:
            print("Loading processed spectra from cache...")
            with open(self.processed_cache, 'rb') as f:
                return pickle.load(f)
        
        print("Processing LRG sample...")
        
        # Get LRG sample
        lrg_sample = self.get_lrg_sample(max_objects, z_min, z_max)
        
        processed_spectra = {}
        
        for i, objid in enumerate(lrg_sample['objid']):
            if i % 100 == 0:
                print(f"Processing {i+1}/{len(lrg_sample['objid'])}: {objid}")
            
            plate = lrg_sample['plate'][i]
            mjd = lrg_sample['mjd'][i]
            fiberid = lrg_sample['fiberid'][i]
            redshift = lrg_sample['z'][i]
            
            # Download spectrum
            spec_file = self.download_spectrum(plate, mjd, fiberid, objid)
            
            if spec_file is None:
                continue
            
            # Load spectrum
            spectrum_data = self.load_spectrum(spec_file)
            
            if spectrum_data is None:
                continue
            
            # Process to rest-frame
            processed = self.process_spectrum(spectrum_data, redshift)
            
            if processed is not None:
                processed_spectra[objid] = processed
            else:
                print(f"Failed to process {objid}")
        
        # Cache processed spectra
        with open(self.processed_cache, 'wb') as f:
            pickle.dump(processed_spectra, f)
        
        print(f"Processed {len(processed_spectra)} spectra successfully")
        return processed_spectra
    
    def plot_sample_spectra(self, processed_spectra, n_plot=5):
        """Plot a sample of processed spectra"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        objids = list(processed_spectra.keys())[:n_plot]
        
        for i, objid in enumerate(objids):
            spec = processed_spectra[objid]
            wave = spec['wavelength']
            flux = spec['flux']
            
            # Normalize for plotting
            median_flux = np.nanmedian(flux)
            if median_flux > 0:
                flux_norm = flux / median_flux + i * 2
                ax.plot(wave, flux_norm, alpha=0.7, label=f"{objid} (z={spec['redshift']:.3f})")
        
        ax.set_xlabel('Rest-frame Wavelength (Å)')
        ax.set_ylabel('Normalized Flux + offset')
        ax.set_title('Sample LRG Spectra (Rest-frame)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Mark common spectral features
        features = {
            'Ca H&K': 3934,
            'G-band': 4300,
            'Mg b': 5175,
            'D4000': 4000
        }
        
        for name, wave_feat in features.items():
            ax.axvline(wave_feat, color='red', alpha=0.5, linestyle='--')
            ax.text(wave_feat, ax.get_ylim()[1]*0.9, name, rotation=90, 
                   verticalalignment='top', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    def clear_cache(self):
        """Clear all cached data"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            print("Cache cleared")

# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = SDSSLRGProcessor(cache_dir='./sdss_lrg_cache')
    
    # Process LRG sample (will use cache on subsequent runs)
    processed_spectra = processor.process_lrg_sample(max_objects=10)
    
    # Plot sample spectra
    if processed_spectra:
        processor.plot_sample_spectra(processed_spectra)
        
        # Access the common wavelength grid and processed spectra
        wave_grid = processor.rest_wave_grid
        print(f"Common wavelength grid: {wave_grid.min():.1f} - {wave_grid.max():.1f} Å")
        print(f"Number of wavelength points: {len(wave_grid)}")
        print(f"Successfully processed {len(processed_spectra)} spectra")
        
        # Example: Stack spectra
        flux_stack = []
        for objid, spec in processed_spectra.items():
            flux_stack.append(spec['flux'])
        
        if flux_stack:
            flux_array = np.array(flux_stack)
            median_stack = np.nanmedian(flux_array, axis=0)
            
            plt.figure(figsize=(10, 6))
            plt.plot(wave_grid, median_stack)
            plt.xlabel('Rest-frame Wavelength (Å)')
            plt.ylabel('Median Flux')
            plt.title('Median Stacked LRG Spectrum')
            plt.grid(True, alpha=0.3)
            plt.show()
    
    # To force reprocessing (ignoring cache):
    # processed_spectra = processor.process_lrg_sample(force_reprocess=True)
    
    # To clear cache:
    # processor.clear_cache()

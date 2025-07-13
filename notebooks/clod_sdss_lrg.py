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
        
        # Common rest-frame wavelength grid (Angstroms)
        self.rest_wave_grid = np.arange(3000, 8000, 1.0)
        
        # SDSS base URL for spectra
        self.base_url = f"https://data.sdss.org/sas/{dr}/eboss/spectro/redux/"
        
        # Cache files
        self.spectra_cache = self.cache_dir / 'spectra_cache.pkl'
        self.processed_cache = self.cache_dir / 'processed_spectra.pkl'
        
    def get_lrg_sample(self, max_objects=100, z_min=0.15, z_max=0.7):
        """
        Get a sample of LRG objects. In practice, you'd query SDSS database.
        This creates a mock sample for demonstration.
        """
        # Mock LRG sample - replace with actual SDSS query
        np.random.seed(42)
        n_obj = min(max_objects, 50)  # Limit for demo
        
        sample = {
            'plate': np.random.randint(3586, 10000, n_obj),
            'mjd': np.random.randint(51608, 59000, n_obj),
            'fiberid': np.random.randint(1, 640, n_obj),
            'z': np.random.uniform(z_min, z_max, n_obj),
            'objid': [f"lrg_{i:04d}" for i in range(n_obj)]
        }
        
        return sample
    
    def get_spectrum_url(self, plate, mjd, fiberid):
        """Construct URL for SDSS spectrum file"""
        run2d = "v5_13_0"  # Common run2d version for DR16
        filename = f"spec-{plate:04d}-{mjd}-{fiberid:04d}.fits"
        url = f"{self.base_url}{run2d}/spectra/{plate:04d}/{filename}"
        return url
    
    def download_spectrum(self, plate, mjd, fiberid, objid):
        """Download and cache a single spectrum"""
        cache_file = self.cache_dir / f"spec_{plate:04d}_{mjd}_{fiberid:04d}.fits"
        
        # Check if already cached
        if cache_file.exists():
            print(f"Loading cached spectrum for {objid}")
            return str(cache_file)
        
        # Download spectrum
        url = self.get_spectrum_url(plate, mjd, fiberid)
        print(f"Downloading spectrum for {objid}: {url}")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save to cache
            with open(cache_file, 'wb') as f:
                f.write(response.content)
            
            print(f"Successfully downloaded and cached {objid}")
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
                
                # Calculate errors from inverse variance
                error = np.where(ivar > 0, 1.0/np.sqrt(ivar), np.inf)
                
                return {
                    'wavelength': wave,
                    'flux': flux,
                    'error': error,
                    'ivar': ivar
                }
        except Exception as e:
            print(f"Error loading spectrum {file_path}: {e}")
            return None
    
    def process_spectrum(self, spectrum_data, redshift):
        """Process spectrum to rest-frame and common wavelength grid"""
        wave = spectrum_data['wavelength']
        flux = spectrum_data['flux']
        error = spectrum_data['error']
        
        # Convert to rest-frame wavelength
        rest_wave = wave / (1 + redshift)
        
        # Apply flux correction for redshift
        rest_flux = flux * (1 + redshift)
        rest_error = error * (1 + redshift)
        
        # Create mask for valid data
        valid_mask = (
            np.isfinite(rest_flux) & 
            np.isfinite(rest_error) & 
            (rest_error > 0) & 
            (rest_wave >= self.rest_wave_grid.min()) & 
            (rest_wave <= self.rest_wave_grid.max())
        )
        
        if np.sum(valid_mask) < 10:
            print("Not enough valid data points for interpolation")
            return None
        
        # Interpolate to common wavelength grid
        try:
            # Use cubic spline interpolation for flux and error
            f_interp = interp1d(
                rest_wave[valid_mask], 
                rest_flux[valid_mask], 
                kind='cubic',
                bounds_error=False,
                fill_value=np.nan
            )
            
            e_interp = interp1d(
                rest_wave[valid_mask], 
                rest_error[valid_mask], 
                kind='cubic',
                bounds_error=False,
                fill_value=np.nan
            )
            
            # Use linear interpolation for inverse variance (more appropriate for weights)
            ivar_interp = interp1d(
                rest_wave[valid_mask],
                spectrum_data['ivar'][valid_mask],
                kind='linear',
                bounds_error=False,
                fill_value=0.0  # Use 0 for invalid regions (infinite error)
            )
            
            # Interpolate to common grid
            interp_flux = f_interp(self.rest_wave_grid)
            interp_error = e_interp(self.rest_wave_grid)
            interp_ivar = ivar_interp(self.rest_wave_grid)
            
            return {
                'wavelength': self.rest_wave_grid.copy(),
                'flux': interp_flux,
                'error': interp_error,
                'ivar': interp_ivar,
                'redshift': redshift
            }
            
        except Exception as e:
            print(f"Error interpolating spectrum: {e}")
            return None
    
    def process_lrg_sample(self, max_objects=100, force_reprocess=False):
        """Process entire LRG sample with caching"""
        
        # Check if processed data exists
        if self.processed_cache.exists() and not force_reprocess:
            print("Loading processed spectra from cache...")
            with open(self.processed_cache, 'rb') as f:
                return pickle.load(f)
        
        print("Processing LRG sample...")
        
        # Get LRG sample
        lrg_sample = self.get_lrg_sample(max_objects)
        
        processed_spectra = {}
        
        for i, objid in enumerate(lrg_sample['objid']):
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
                print(f"Successfully processed {objid}")
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

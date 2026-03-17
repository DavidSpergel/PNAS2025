# This code is used to plot the power spectrum of the image and the wavelet power spectrum
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import pywt

from matplotlib.colors import LogNorm

# Set range for fitting the power spectrum in pixel units
lmin_fit = 4
lmax_fit = 128
# Increase font size for axis labels and title
plt.rcParams.update({'font.size': 20})  # Increase base font size
plt.rcParams['axes.labelsize'] = 20     # Increase axis label size
plt.rcParams['axes.titlesize'] = 20     # Increase title size
gamma_est = 2.5 # estimated value for error bars

# Read the FITS file
with fits.open('/Users/dspergel/Simons Foundation Dropbox/David Spergel/AstroGMT/Wavelet Paper Code/casa_4.0-6.0keV.fits') as hdul:
    data = hdul[0].data

    # --- Plot and save original image as high-res PDF ---
    fig_img = plt.figure(figsize=(10, 8))
    data[data < 0] = 0
    plt.close()  # Close any existing figures
    fig_img = plt.figure(figsize=(10, 8))
    
    # Get pixel scale from FITS header (assumed to be in degrees)
    pixel_scale = abs(hdul[0].header.get('CDELT1', 1)) *60*60 # Convert degrees to arcseconds

    # Calculate extent in arcminutes
    height, width = data.shape
    extent = [-width/2 * pixel_scale, width/2 * pixel_scale, 
              -height/2 * pixel_scale, height/2 * pixel_scale]

    plt.xlim(-200, 200)
    plt.ylim(-200, 200)
    
    plt.imshow(data, norm=LogNorm(), cmap='gray', extent=extent)
    plt.colorbar(label='Intensity')

    title = hdul[0].header.get('OBJECT', 'Cas A')
    # plt.title(title)
    # plt.xlabel('Pixels')
    # plt.ylabel('Pixels')
    plt.tight_layout()
    fig_img.savefig("original_image.pdf", format="pdf", dpi=600)  # Higher resolution PDF export
    plt.show()

    # --- Wavelet Transform and Power Spectrum ---
    # Define scale range for wavelet transform using powers of 2
    scales = 2**np.arange(0, np.log2(min(height, width))-1)
    wavelet_power = []
    
    # Calculate wavelet power spectrum
    for scale in scales:
        # Create Mexican hat wavelet filter
        x = np.linspace(-scale*4, scale*4, int(scale*8))
        y = x[:, np.newaxis]
        wavelet = np.exp(-(x**2 + y**2)/(2*scale**2)) * (1 - (x**2 + y**2)/(2*scale**2))
        # L2 normalize the wavelet
        wavelet = wavelet / np.sqrt(np.sum(wavelet**2))      
        # Convolve with image
        conv = np.abs(np.fft.ifft2(np.fft.fft2(data) * np.fft.fft2(wavelet, data.shape)))
        wavelet_power.append(np.mean(conv**2))

    # --- Plot and save wavelet power spectrum as high-res PDF ---
    fig_wave = plt.figure(figsize=(10, 8))
    nmodes = (min(height, width)/scales)**gamma_est
    sig_wavelet_power = np.array(wavelet_power)/np.sqrt(nmodes/2)
    mask = (scales >= lmin_fit) & (scales <= lmax_fit)
    scales_arcsec = scales * abs(hdul[0].header.get('CDELT1', 1)) * 60 * 60

    plt.errorbar(scales_arcsec, wavelet_power, yerr=sig_wavelet_power, fmt='k.', capsize=3, label='Wavelet Power')
    plt.xscale('log')
    plt.yscale('log')

    # Perform weighted linear fit to the log-log data, using sig_wavelet_power as weights
    log_scales = np.log10(scales_arcsec[mask])
    log_power = np.log10(np.array(wavelet_power)[mask])
    log_errors = sig_wavelet_power[mask] / np.array(wavelet_power)[mask] / np.log(10)

    fit, cov = np.polyfit(log_scales, log_power, 1, w=1/log_errors, cov=True)
    slope = fit[0]
    slope_err = np.sqrt(cov[0, 0])

    # Plot fit line over full range for comparison
    plt.loglog(scales_arcsec, 10**(slope * np.log10(scales_arcsec) + fit[1]), 'r-',
               label=f'Slope: {slope:.2f}')
    plt.text(0.05, 0.95, f'Best fit slope: {slope:.2f} ± {slope_err:.2f}',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel('Wavelet Scale (arcsec)')
    plt.ylabel('Wavelet Power')
    # plt.title('Wavelet Power Spectrum')
    plt.tight_layout()
    fig_wave.savefig("fig4c.pdf", format="pdf", dpi=600)  # Higher resolution PDF export
    plt.show()

    # Compute wavelet power spectrum for central 1000x1000 region

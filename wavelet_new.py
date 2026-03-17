# This code is used to plot the power spectrum of the image and the wavelet power spectrum
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import pywt

from matplotlib.colors import LogNorm


# Set range for fitting the power spectrum in pixel units
lmin_fit = 2
lmax_fit = 128
# Increase font size for axis labels and title
plt.rcParams.update({'font.size': 20})  # Increase base font size
plt.rcParams['axes.labelsize'] = 20     # Increase axis label size
plt.rcParams['axes.titlesize'] = 20     # Increase title size

# Read the FITS file
with fits.open('/Users/dspergel/Simons Foundation Dropbox/David Spergel/Wavelet Paper Code/casa_0.5-1.5keV.fits') as hdul:
#with fits.open('casa_4.0-6.0keV.fits') as hdul:
    data = hdul[0].data
    # Create figure for original image
    plt.figure(figsize=(10, 8))
      # Set minimum threshold

 #   data = np.log10(data+1)
    # Apply log scaling
    data[data < 0] = 0
    plt.close()  # Close any existing figures
    fig1 = plt.figure(figsize=(10, 8))
    
    # Get pixel scale from FITS header (assumed to be in degrees)
    pixel_scale = abs(hdul[0].header.get('CDELT1', 1)) *60*60 # Convert degrees to arcseconds
    
    # Calculate extent in arcminutes
    height, width = data.shape
    extent = [-width/2 * pixel_scale, width/2 * pixel_scale, 
              -height/2 * pixel_scale, height/2 * pixel_scale]
    # Set plot limits to ±200 arcseconds
    # Vary the limits to match the image size

    plt.xlim(-200, 200)
    plt.ylim(-200, 200)
    
    plt.imshow(data, norm=LogNorm(), cmap='gray', extent=extent)

    plt.colorbar(label='Intensity')
    # Generate Gaussian random field with k^-1.5 power spectrum
    freq_x = np.fft.fftfreq(width)
    freq_y = np.fft.fftfreq(height)
    kx, ky = np.meshgrid(freq_x, freq_y)
    k = np.sqrt(kx**2 + ky**2)
    k[0,0] = 1.0  # Avoid division by zero
    
    # Generate power spectrum P(k) ~ k^(-1.5)
    power = k**(-1.5)
    
    # Generate complex Gaussian random numbers
    random_complex = (np.random.normal(size=(height, width)) + 
                     1j * np.random.normal(size=(height, width))) / np.sqrt(2)
    
    # Apply power spectrum
    field_k = random_complex * np.sqrt(power)
    
    # Inverse FFT to get real space field
    field = np.real(np.fft.ifft2(field_k))
    
    # Normalize field
    field = (field - np.mean(field)) / np.std(field)
    
    # Overwrite data with generated field
    #data = field

    # Add title (using object name from header if available)
    title = hdul[0].header.get('OBJECT', 'Cas A')
    #plt.title(title)
        
    # Add axis labels
   # plt.xlabel('Pixels')
   # plt.ylabel('Pixels')
    # Save the original image as an EPS file at high resolution
    fig1.savefig('original_image.eps', format='eps', dpi=600, bbox_inches='tight')
    plt.show()

    # Wavelet transform
    # Define scale range for wavelet transform using powers of 2
    # The scale is the size of the wavelet in pixels
    # Vary the range to match the image size
    scales = 2**np.arange(0, np.log2(min(height,width))-1)  # Gives [1,2,4,8,...,2048]
    # Initialize list for wavelet power
    wavelet_power = []
    
 # Calculate wavelet power spectrum
 # The wavelet is a Ricker wavelet with an L2 normalization
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


    # Plot wavelet power spectrum
    fig2 = plt.figure(figsize=(10, 8))

    # Fit power law only to scales between 8 and 256
    mask = (scales >= lmin_fit) & (scales <= lmax_fit)
    scales = scales*abs(hdul[0].header.get('CDELT1', 1)) *60*60

    plt.loglog(scales, wavelet_power, 'k.')
    
    fit = np.polyfit(np.log10(scales[mask]), np.log10(np.array(wavelet_power)[mask]), 1)
    
    # Plot fit line over full range for comparison
    
    plt.loglog(scales, 10**(fit[0]*np.log10(scales) + fit[1]), 'r-',
              label=f'Slope: {fit[0]:.2f}')
    plt.text(0.05, 0.95, f'Best fit slope: {fit[0]:.2f}', 
             transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel('Wavelet Scale (arcsec)')
    plt.ylabel('Wavelet Power')
   # plt.title('Wavelet Power Spectrum')
    # Save as EPS with high resolution
    fig2.savefig('wavelet_power_spectrum.eps', format='eps', dpi=600, bbox_inches='tight')
    plt.show()
    # Calculate separate wavelet power spectra for different orientations
    wavelet_power_h = []  # horizontal
    wavelet_power_v = []  # vertical 
    wavelet_power_d = []  # diagonal

    for scale in scales:
        # Create oriented wavelets
        x = np.linspace(-scale*4, scale*4, int(scale*8))
        y = x[:, np.newaxis]
        
        # Horizontal wavelet (varies in x direction)
        wavelet_h = np.exp(-y**2/(2*scale**2)) * np.exp(-x**2/(2*scale**2)) * (1 - x**2/(scale**2))
        wavelet_h = wavelet_h / np.sqrt(np.sum(wavelet_h**2))
        
        # Vertical wavelet (varies in y direction)
        wavelet_v = np.exp(-x**2/(2*scale**2)) * np.exp(-y**2/(2*scale**2)) * (1 - y**2/(scale**2))
        wavelet_v = wavelet_v / np.sqrt(np.sum(wavelet_v**2))
        
        # Diagonal wavelet (varies in both directions)
        wavelet_d = np.exp(-(x**2 + y**2)/(2*scale**2)) * (2*x*y/(2*scale**2))
        wavelet_d = wavelet_d / np.sqrt(np.sum(wavelet_d**2))

        # Compute power for each orientation
        conv_h = np.abs(np.fft.ifft2(np.fft.fft2(data) * np.fft.fft2(wavelet_h, data.shape)))
        conv_v = np.abs(np.fft.ifft2(np.fft.fft2(data) * np.fft.fft2(wavelet_v, data.shape)))
        conv_d = np.abs(np.fft.ifft2(np.fft.fft2(data) * np.fft.fft2(wavelet_d, data.shape)))
        
        wavelet_power_h.append(np.mean(conv_h**2))
        wavelet_power_v.append(np.mean(conv_v**2))
        wavelet_power_d.append(np.mean(conv_d**2))

    # Plot oriented wavelet power spectra
    fig3 = plt.figure(figsize=(12, 8))
    
    # Convert scales to arcseconds
    scales_arcsec = scales*abs(hdul[0].header.get('CDELT1', 1)) *60*60

    # Plot data points
    plt.loglog(scales_arcsec, wavelet_power_h, 'b.', label='Horizontal', alpha=0.5)
    plt.loglog(scales_arcsec, wavelet_power_v, 'r.', label='Vertical', alpha=0.5)
    plt.loglog(scales_arcsec, wavelet_power_d, 'g.', label='Diagonal', alpha=0.5)

    # Fit power laws
    mask = (scales >= lmin_fit) & (scales <= lmax_fit)
    
    fit_h = np.polyfit(np.log10(scales_arcsec[mask]), np.log10(np.array(wavelet_power_h)[mask]), 1)
    fit_v = np.polyfit(np.log10(scales_arcsec[mask]), np.log10(np.array(wavelet_power_v)[mask]), 1)
    fit_d = np.polyfit(np.log10(scales_arcsec[mask]), np.log10(np.array(wavelet_power_d)[mask]), 1)

    # Plot fit lines
    plt.loglog(scales_arcsec, 10**(fit_h[0]*np.log10(scales_arcsec) + fit_h[1]), 'b-',
               label=f'Horizontal slope: {fit_h[0]:.2f}')
    plt.loglog(scales_arcsec, 10**(fit_v[0]*np.log10(scales_arcsec) + fit_v[1]), 'r-',
               label=f'Vertical slope: {fit_v[0]:.2f}')
    plt.loglog(scales_arcsec, 10**(fit_d[0]*np.log10(scales_arcsec) + fit_d[1]), 'g-',
               label=f'Diagonal slope: {fit_d[0]:.2f}')

    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel('Wavelet Scale (arcsec)')
    plt.ylabel('Wavelet Power')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    # Save as EPS with high resolution
    fig3.savefig('wavelet_oriented_power_spectrum.eps', format='eps', dpi=600, bbox_inches='tight')
    plt.show()

    # Compute wavelet power spectrum using Mexican hat wavelets at different scales
    scales_mh = 2**np.arange(1, int(np.log2(min(data.shape))-1))
    
    # Initialize power arrays
    mh_power_h = []
    mh_power_v = []
    mh_power_d = []
    mh_four_h = []
    mh_four_v = []
    mh_four_d = []

    # Calculate power at each scale
    for scale in scales_mh:
        # Create coordinate grid
        x = np.linspace(-scale*4, scale*4, int(scale*8))
        y = x[:, np.newaxis]
        
        # Horizontal Mexican hat wavelet
        wavelet_h = np.exp(-x**2/(2*scale**2)) * np.exp(-y**2/(2*scale**2)) * (1 - x**2/(scale**2))
        wavelet_h = wavelet_h / np.sqrt(np.sum(wavelet_h**2))
        
        # Vertical Mexican hat wavelet
        wavelet_v = np.exp(-x**2/(2*scale**2)) * np.exp(-y**2/(2*scale**2)) * (1 - y**2/(scale**2))
        wavelet_v = wavelet_v / np.sqrt(np.sum(wavelet_v**2))
        
        # Diagonal Mexican hat wavelet
        wavelet_d = np.exp(-(x**2 + y**2)/(2*scale**2)) * (2*x*y/(2*scale**2))
        wavelet_d = wavelet_d / np.sqrt(np.sum(wavelet_d**2))

        # Compute convolutions
        conv_h = np.abs(np.fft.ifft2(np.fft.fft2(data) * np.fft.fft2(wavelet_h, data.shape)))
        conv_v = np.abs(np.fft.ifft2(np.fft.fft2(data) * np.fft.fft2(wavelet_v, data.shape)))
        conv_d = np.abs(np.fft.ifft2(np.fft.fft2(data) * np.fft.fft2(wavelet_d, data.shape)))

        # Calculate power
        mh_power_h.append(np.mean(conv_h**2))
        mh_power_v.append(np.mean(conv_v**2))
        mh_power_d.append(np.mean(conv_d**2))
        mh_four_h.append(np.mean(conv_h**4))
        mh_four_v.append(np.mean(conv_v**4))
        mh_four_d.append(np.mean(conv_d**4))
        mh_four_d[-1] = mh_four_d[-1]/np.array(mh_power_d[-1])**2
        mh_four_v[-1] = mh_four_v[-1]/np.array(mh_power_v[-1])**2
        mh_four_h[-1] = mh_four_h[-1]/np.array(mh_power_h[-1])**2
    # Plot Mexican hat wavelet power spectra
    fig4 = plt.figure(figsize=(12, 8))
    scales_mh_arcsec = scales_mh * pixel_scale

    # Fit power laws for Mexican hat wavelets
    lmin_fit_mh = 4
    lmax_fit_mh = 2**(len(scales_mh)-2)
    mask_mh = (scales_mh >= lmin_fit_mh) & (scales_mh <= lmax_fit_mh)
    
    fit_mh_h = np.polyfit(np.log10(scales_mh_arcsec[mask_mh]), np.log10(np.array(mh_power_h)[mask_mh]), 1)
    fit_mh_v = np.polyfit(np.log10(scales_mh_arcsec[mask_mh]), np.log10(np.array(mh_power_v)[mask_mh]), 1)
    fit_mh_d = np.polyfit(np.log10(scales_mh_arcsec[mask_mh]), np.log10(np.array(mh_power_d)[mask_mh]), 1)

    # Plot fit lines
    plt.loglog(scales_mh_arcsec, 10**(fit_mh_h[0]*np.log10(scales_mh_arcsec) + fit_mh_h[1]), 'b-',
               label=f'Horizontal slope: {fit_mh_h[0]:.2f}')
    plt.loglog(scales_mh_arcsec, 10**(fit_mh_v[0]*np.log10(scales_mh_arcsec) + fit_mh_v[1]), 'r-',
               label=f'Vertical slope: {fit_mh_v[0]:.2f}')
    plt.loglog(scales_mh_arcsec, 10**(fit_mh_d[0]*np.log10(scales_mh_arcsec) + fit_mh_d[1]), 'g-',
               label=f'Diagonal slope: {fit_mh_d[0]:.2f}')

    plt.loglog(scales_mh_arcsec, mh_power_h, 'b.', label='Mexican Hat Horizontal', alpha=0.5)
    plt.loglog(scales_mh_arcsec, mh_power_v, 'r.', label='Mexican Hat Vertical', alpha=0.5)
    plt.loglog(scales_mh_arcsec, mh_power_d, 'g.', label='Mexican Hat Diagonal', alpha=0.5)

    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel('Mexican Hat Wavelet Scale (arcsec)')
    plt.ylabel('Mexican Hat Wavelet Power')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    # Save as EPS with high resolution
    fig4.savefig('mexican_hat_wavelet_power_spectrum.eps', format='eps', dpi=600, bbox_inches='tight')
    plt.show()
    fit_mh_h = np.polyfit(np.log10(scales_mh_arcsec[mask_mh]), np.log10(np.array(mh_four_h)[mask_mh]), 1)
    fit_mh_v = np.polyfit(np.log10(scales_mh_arcsec[mask_mh]), np.log10(np.array(mh_four_v)[mask_mh]), 1)
    fit_mh_d = np.polyfit(np.log10(scales_mh_arcsec[mask_mh]), np.log10(np.array(mh_four_d)[mask_mh]), 1)

    # Plot fit lines
    #plt.loglog(scales_mh_arcsec, 10**(fit_mh_h[0]*np.log10(scales_mh_arcsec) + fit_mh_h[1]), 'b-',
    #           label=f'Horizontal slope: {fit_mh_h[0]:.2f}')
    #plt.loglog(scales_mh_arcsec, 10**(fit_mh_v[0]*np.log10(scales_mh_arcsec) + fit_mh_v[1]), 'r-',
    #           label=f'Vertical slope: {fit_mh_v[0]:.2f}')
    #plt.loglog(scales_mh_arcsec, 10**(fit_mh_d[0]*np.log10(scales_mh_arcsec) + fit_mh_d[1]), 'g-',
    #           label=f'Diagonal slope: {fit_mh_d[0]:.2f}')

    fig5 = plt.figure(figsize=(12, 8))
    plt.loglog(scales_mh_arcsec, mh_four_h, 'b.', label='Mexican Hat Horizontal', alpha=0.5)
    plt.loglog(scales_mh_arcsec, mh_four_v, 'r.', label='Mexican Hat Vertical', alpha=0.5)
    plt.loglog(scales_mh_arcsec, mh_four_d, 'g.', label='Mexican Hat Diagonal', alpha=0.5)

    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel('Mexican Hat Wavelet Scale (arcsec)', fontsize=12)
    plt.ylabel('Mexican Hat Wavelet Kurtosis', fontsize=12)
    plt.legend(bbox_to_anchor=(0.05, 0.5), loc='upper left', fontsize=10)
    # Save as EPS with high resolution
    fig5.savefig('mexican_hat_wavelet_kurtosis.eps', format='eps', dpi=600, bbox_inches='tight')
    plt.show()
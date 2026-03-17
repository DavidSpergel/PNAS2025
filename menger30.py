import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': plt.rcParams['font.size'] * 2})
import matplotlib.pyplot as plt
import os

# Create directory for plots if it doesn't exist
if not os.path.exists('menger_plots'):
    os.makedirs('menger_plots')

def generate_menger_sponge(level, size=1.0, center=(0, 0, 0)):
    """
    Generate a 3D Menger sponge by recursively subdividing cubes
    Returns a list of (x, y, z, size) tuples representing the remaining cubes
    """
    if level == 0:
        return [(*center, size)]
    
    cubes = []
    cube_size = size / 3.0
    
    # Generate 27 subcubes, but remove the middle ones (creating holes)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Skip middle cubes to create the sponge pattern
                middle_count = (i == 1) + (j == 1) + (k == 1)
                if middle_count >= 2:  # Remove cubes that are middle in 2 or 3 dimensions
                    continue
                
                # Calculate position of subcube
                x = center[0] + (i - 1) * cube_size
                y = center[1] + (j - 1) * cube_size
                z = center[2] + (k - 1) * cube_size
                
                # Recursively generate smaller cubes
                subcubes = generate_menger_sponge(level - 1, cube_size, (x, y, z))
                cubes.extend(subcubes)
    
    return cubes

def project_along_inclined_line(cubes, resolution=2048, angle_deg=30):
    """
    Project 3D Menger sponge onto a 2D plane by integrating density along lines
    inclined at angle_deg degrees to the principal axes
    """
    if not cubes:
        return np.zeros((resolution, resolution))
    
    # Convert angle to radians
    angle_rad = np.radians(angle_deg)
    
    # Direction vector for the inclined line (30 degrees from each axis)
    # We'll use a direction that makes equal angles with x, y, z axes
    cos_angle = np.cos(angle_rad)
    direction = np.array([cos_angle, cos_angle, cos_angle])
    direction = direction / np.linalg.norm(direction)  # Normalize
    
    # Create two orthogonal vectors perpendicular to the direction
    # First perpendicular vector
    perp1 = np.array([1, -1, 0])
    perp1 = perp1 / np.linalg.norm(perp1)
    
    # Second perpendicular vector (cross product)
    perp2 = np.cross(direction, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)
    
    # Project all cube centers onto the perpendicular plane
    projected_coords = []
    for x, y, z, size in cubes:
        pos = np.array([x, y, z])
        # Project onto the plane defined by perp1 and perp2
        coord1 = np.dot(pos, perp1)
        coord2 = np.dot(pos, perp2)
        projected_coords.append((coord1, coord2, size))
    
    # Find bounds of projected coordinates
    coord1_vals = [coord[0] for coord in projected_coords]
    coord2_vals = [coord[1] for coord in projected_coords]
    
    coord1_min, coord1_max = min(coord1_vals), max(coord1_vals)
    coord2_min, coord2_max = min(coord2_vals), max(coord2_vals)
    
    # Add padding
    padding = 0.05 * max(coord1_max - coord1_min, coord2_max - coord2_min)
    coord1_min -= padding
    coord1_max += padding
    coord2_min -= padding
    coord2_max += padding
    
    # Create 2D image
    image = np.zeros((resolution, resolution))
    
    # For each cube, determine its contribution to the 2D projection
    for coord1, coord2, size in projected_coords:
        # Convert to pixel coordinates
        i = int((coord2 - coord2_min) / (coord2_max - coord2_min) * resolution)
        j = int((coord1 - coord1_min) / (coord1_max - coord1_min) * resolution)
        
        # Calculate cube projection size in pixels
        size_pixels = int(size / max(coord1_max - coord1_min, coord2_max - coord2_min) * resolution)
        size_pixels = max(1, size_pixels)  # Ensure at least 1 pixel
        
        # Add density around the projected point
        for di in range(-size_pixels//2, size_pixels//2 + 1):
            for dj in range(-size_pixels//2, size_pixels//2 + 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < resolution and 0 <= nj < resolution:
                    # Weight by cube size to represent integration along the inclined direction
                    weight = size
                    image[ni, nj] += weight
    
    return image, coord1_min, coord1_max, coord2_min, coord2_max

# Generate Menger sponge at different levels
level = 6  # Adjust level as needed (higher = more detail, slower computation)
print(f"Generating Menger sponge at level {level}...")

cubes = generate_menger_sponge(level)
print(f"Generated {len(cubes)} cubes")

# Project along inclined lines (30 degrees to principal axes)
resolution = 2048
angle = 30
print(f"Projecting along lines inclined at {angle} degrees to principal axes...")
projected_image, c1_min, c1_max, c2_min, c2_max = project_along_inclined_line(cubes, resolution, angle)

# Display the result without axis labels
plt.figure(figsize=(10, 8))
plt.imshow(projected_image, cmap='hot', origin='lower', 
           extent=[c1_min, c1_max, c2_min, c2_max])
plt.colorbar(label='Integrated Density')
#plt.title(f'Menger Sponge (Level {level}) - Projection along {angle}° Inclined Lines')
#plt.xlabel('Projected Coordinate 1')
#plt.ylabel('Projected Coordinate 2')
plt.savefig(f'menger_plots/menger_sponge_level_{level}_inclined_{angle}deg.pdf', dpi=600, bbox_inches='tight')
plt.show()

print(f"Projection complete. Non-zero pixels: {np.count_nonzero(projected_image)}")
print(f"Maximum density: {np.max(projected_image):.3f}")
print(f"Total integrated density: {np.sum(projected_image):.3f}")
def box_counting_dimension(image, threshold=0, min_box_size=2, max_box_size=None):
    """
    Calculate fractal dimension using box counting method
    """
    if max_box_size is None:
        max_box_size = min(image.shape) // 4
    
    box_sizes = []
    box_counts = []
    
    # Convert image to binary based on threshold
    binary_image = (image > threshold).astype(int)
    
    # Try different box sizes
    current_size = min_box_size
    while current_size <= max_box_size:
        box_sizes.append(current_size)
        
        # Count boxes that contain part of the fractal
        count = 0
        for i in range(0, binary_image.shape[0], current_size):
            for j in range(0, binary_image.shape[1], current_size):
                box = binary_image[i:i+current_size, j:j+current_size]
                if np.any(box > 0):
                    count += 1
        
        box_counts.append(count)
        current_size *= 2
    
    # Fit line to log-log plot to get dimension
    if len(box_sizes) < 2:
        return np.nan, box_sizes, box_counts
    
    log_sizes = np.log(box_sizes)
    log_counts = np.log(box_counts)
    
    # Linear regression
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    dimension = -coeffs[0]  # Negative slope gives the dimension
    
    return dimension, box_sizes, box_counts

# Analyze fractal dimension at different threshold levels
print("\nAnalyzing fractal dimension at different threshold levels...")

# Define threshold levels as percentiles of the image
thresholds = [0, 0.1, 0.25, 0.5, 0.75, 0.9]
max_val = np.max(projected_image)
threshold_values = [t * max_val for t in thresholds]

dimensions = []
threshold_labels = []

plt.figure(figsize=(15, 10))

for i, (threshold_frac, threshold_val) in enumerate(zip(thresholds, threshold_values)):
    dimension, box_sizes, box_counts = box_counting_dimension(projected_image, threshold_val)
    dimensions.append(dimension)
    threshold_labels.append(f"{threshold_frac:.1f}")
    
    print(f"Threshold {threshold_frac:.1f} (value {threshold_val:.3f}): Dimension = {dimension:.3f}")
    
    # Plot box counting results
    if len(box_sizes) > 1:
        plt.subplot(2, 3, i+1)
        plt.loglog(box_sizes, box_counts, 'bo-')
        
        # Plot fitted line
        log_sizes = np.log(box_sizes)
        log_counts = np.log(box_counts)
        fitted_line = np.exp(np.polyval(np.polyfit(log_sizes, log_counts, 1), log_sizes))
        plt.loglog(box_sizes, fitted_line, 'r--', label=f'D = {dimension:.3f}')
        
        plt.xlabel('Box Size')
        plt.ylabel('Box Count')
    #    plt.title(f'Threshold {threshold_frac:.1f} (D = {dimension:.3f})')
     #   plt.legend()
        plt.grid(True)

plt.tight_layout()
plt.savefig(f'menger_plots/menger_sponge_level_{level}_fractal_dimension_.pdf', dpi=600)
plt.show()
plt.close()

# Plot fractal dimension vs threshold
plt.figure(figsize=(10, 8))
plt.plot(threshold_values, dimensions, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Threshold Value')
plt.ylabel('Fractal Dimension')
#plt.title('Fractal Dimension vs Threshold Level')
plt.grid(True)
plt.savefig(f'menger_plots/menger_sponge_level_{level}_fractal_dimension_vs_threshold.pdf', dpi=600)
plt.show()
plt.close()

print(f"\nFractal dimension summary:")
for thresh, dim in zip(threshold_labels, dimensions):
    print(f"Threshold {thresh}: D = {dim:.3f}")
# Compute 2D FFT power spectrum
fft_2d = np.fft.fft2(projected_image)
power_spectrum_2d = np.abs(fft_2d)**2

# Create frequency grids
ny, nx = projected_image.shape
freq_x = np.fft.fftfreq(nx)
freq_y = np.fft.fftfreq(ny)
freq_x_grid, freq_y_grid = np.meshgrid(freq_x, freq_y)

# Calculate radial frequencies
freq_radial = np.sqrt(freq_x_grid**2 + freq_y_grid**2)

# Bin the power spectrum radially
max_freq = 0.5  # Nyquist frequency
freq_bins = np.logspace(np.log10(1/resolution), np.log10(max_freq), 50)
freq_centers = (freq_bins[1:] + freq_bins[:-1]) / 2
radial_power = []

for i in range(len(freq_bins)-1):
    mask = (freq_radial >= freq_bins[i]) & (freq_radial < freq_bins[i+1]) & (freq_radial > 0)
    if np.any(mask):
        radial_power.append(np.mean(power_spectrum_2d[mask]))
    else:
        radial_power.append(0)

radial_power = np.array(radial_power)

# Remove zero power values for fitting
valid_mask = radial_power > 0
freq_centers_valid = freq_centers[valid_mask]
radial_power_valid = radial_power[valid_mask]

# Convert to wavenumber (k = 2π * frequency)
wavenumbers = 2 * np.pi * freq_centers_valid

# Fit power law to middle range of wavenumbers
k_min_fit = wavenumbers[len(wavenumbers)//4]  # Start from 1/4 of the range
k_max_fit = wavenumbers[3*len(wavenumbers)//4]  # End at 3/4 of the range
fit_mask = (wavenumbers >= k_min_fit) & (wavenumbers <= k_max_fit)

if np.sum(fit_mask) > 2:  # Need at least 3 points for fitting
    fft_fit = np.polyfit(np.log10(wavenumbers[fit_mask]), np.log10(radial_power_valid[fit_mask]), 1)
    fft_slope = fft_fit[0]
else:
    fft_slope = np.nan

# Plot FFT power spectrum
plt.figure(figsize=(10, 8))
plt.loglog(wavenumbers, radial_power_valid, 'b.', label='FFT Power')

if not np.isnan(fft_slope):
    plt.loglog(wavenumbers, 10**(fft_fit[0]*np.log10(wavenumbers) + fft_fit[1]), 'r-',
               label=f'Slope: {fft_slope:.2f}')

plt.xlabel('Wavenumber k')
plt.ylabel('Power')
#plt.title('FFT Power Spectrum')
#plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.text(0.05, 0.95, f'Best fit slope: {fft_slope:.2f}', 
         transform=plt.gca().transAxes, 
         bbox=dict(facecolor='white', alpha=0.8))
plt.savefig(f'menger_plots/menger_sponge_level_{level}_fft_power.pdf', dpi=600)
plt.show()
plt.close()

print(f"FFT Power Spectrum Slope: {fft_slope:.3f}")

scales = 2**np.arange(0, np.log2(resolution)) # Gives [1,2,4,8,...,2048]
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
            conv = np.abs(np.fft.ifft2(np.fft.fft2(projected_image) * np.fft.fft2(wavelet, projected_image.shape)))
            wavelet_power.append(np.mean(conv**2))


    # Plot wavelet power spectrum
plt.figure(figsize=(10, 8))
lmin_fit =8
lmax_fit=128
    # Output power vs scale data to file
with open(f'menger_plots/menger_sponge_level_{level}_power_vs_scale.txt', 'w') as f:
        f.write("# Scale\tWavelet Power\n")
        for scale, power in zip(scales, wavelet_power):
            f.write(f"{scale}\t{power}\n")

    # Fit power law only to scales between 8 and 256
mask = (scales >= lmin_fit) & (scales <= lmax_fit)


plt.loglog(scales, wavelet_power, 'k.')
    
fit = np.polyfit(np.log10(scales[mask]), np.log10(np.array(wavelet_power)[mask]), 1)
    
    # Plot fit line over full range for comparison
    # Plot fit line over full range for comparison
plt.loglog(scales, 10**(fit[0]*np.log10(scales) + fit[1]), 'r-',
              label=f'Slope: {fit[0]:.2f}')
#plt.text(0.05, 0.95, f'Best fit slope: {fit[0]:.2f}', 
#            transform=plt.gca().transAxes, 
#             bbox=dict(facecolor='white', alpha=0.8))
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.xlabel('Wavelet Scale')
plt.ylabel('Wavelet Power')
   # plt.title('Wavelet Power Spectrum')
plt.savefig(f'menger_plots/menger_sponge_level_{level}_wavelet_power.pdf', dpi=600)
plt.show()
plt.close()
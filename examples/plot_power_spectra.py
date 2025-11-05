"""

Example on deriving lidar moments from raw IQ data
--------------------------------------------------
"""

import highiq
import matplotlib.pyplot as plt

# Load an example ARM compliant-file
test_file = highiq.io.load_arm_netcdf(highiq.testing.TEST_FILE)
print(test_file)
# Get the particle size distributions
out_ds = highiq.calc.get_psd(test_file)

# Plot the power spectra for a given time and height
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
out_ds["power_spectral_density"].sel(range=350.0, method="nearest").plot(
    ax=ax,
    label="350 meters",
)
out_ds["power_spectral_density"].sel(range=950.0, method="nearest").plot(
    ax=ax,
    label="950 meters",
)
plt.legend()
plt.show()
plt.savefig("power_spectra_example.png")
test_file.close()

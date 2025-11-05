"""

Example on plotting moments from raw IQ data
---------------------------------------------
"""

import highiq
import matplotlib.pyplot as plt

# Load an example ARM compliant-file
test_file = highiq.io.load_arm_netcdf(highiq.testing.TEST_FILE)

# Get the particle size distributions
my_ds = highiq.calc.get_psd(test_file)
my_ds = highiq.calc.get_lidar_moments(my_ds)
print(my_ds)
# Filter dataset based on SNR
my_ds["radial_velocity_max_peak"] = my_ds["radial_velocity_max_peak"].where(
    my_ds.intensity > 1.008
)

# Plot the power spectra for a given time and height
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
my_ds["intensity"].plot(ax=ax[0], y="range")
my_ds["radial_velocity_max_peak"].plot(ax=ax[1], y="range")
plt.show()
plt.savefig("moments_example.png")
test_file.close()

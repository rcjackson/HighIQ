"""

Example on plotting moments from raw IQ data
---------------------------------------------
"""
import highiq
import matplotlib.pyplot as plt

from datetime import datetime

# Load an example ARM compliant-file
test_file = highiq.io.load_arm_netcdf(highiq.testing.TEST_FILE)

# Get the particle size distributions
my_ds = highiq.calc.get_psd(test_file)
my_ds = highiq.calc.get_lidar_moments(test_file)
print(my_ds.attrs)
# Filter dataset based on SNR
my_ds["doppler_velocity_max_peak"] = my_ds["doppler_velocity_max_peak"].where(my_ds.snr > 1)

# Plot the power spectra for a given time and height
my_time = datetime(2017, 8, 4, 0, 40, 59)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
my_ds["snr"].plot(ax=ax[0], y='range')
my_ds["doppler_velocity_max_peak"].plot(ax=ax[1], y='range')
plt.show()

test_file.close()

"""

Example on deriving lidar moments from raw IQ data
--------------------------------------------------
"""
import highiq
import matplotlib.pyplot as plt

from datetime import datetime

# Load an example ARM compliant-file
test_file = highiq.io.load_arm_netcdf(highiq.testing.TEST_FILE)

# Get the particle size distributions
my_ds = highiq.calc.get_psd(test_file)

# Plot the power spectra for a given time and height
my_time = datetime(2017, 8, 4, 0, 40, 59)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
my_ds["power_spectra_normed_interp"].sel(time=my_time, range=350., method='nearest').plot(ax=ax[0])
my_ds["power_spectra_normed_interp"].sel(time=my_time, range=950., method='nearest').plot(ax=ax[1])
plt.show()

test_file.close()
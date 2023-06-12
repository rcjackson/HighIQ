"""

Example on deriving lidar moments from raw IQ data
--------------------------------------------------
"""
import highiq
import matplotlib.pyplot as plt

from datetime import datetime

# Load an example ARM compliant-file
test_file = highiq.io.load_arm_netcdf('/Users/rjackson/Downloads/sgpdlacfC1.a1.20170804.000113.nc.v0')
print(test_file)
# Get the particle size distributions
out_ds = highiq.calc.get_psd(test_file)

# Plot the power spectra for a given time and height
my_time = datetime(2017, 8, 4, 0, 40, 59)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
out_ds["power_spectral_density"].sel(time=my_time, range=350., method='nearest').plot(ax=ax[0])
out_ds["power_spectral_density"].sel(time=my_time, range=950., method='nearest').plot(ax=ax[1])
plt.show()

test_file.close()

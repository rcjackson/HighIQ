"""
=============================================================
Example on processing moments from ARM Doppler Lidar ACF data
=============================================================
"""

import highiq
import matplotlib.pyplot as plt

# Load the data. This uses a module in the Atmospheric Community
# Toolkit to ensure that the input acf data are always
# entered into a 3D array of (time, nsamples, complex)

my_ds = highiq.io.load_arm_netcdf(highiq.testing.TEST_FILE)
my_ds_processed = highiq.calc.get_psd(my_ds)

# For plotting the spectra, simply use xarray's built in functionality
my_ds_processed.power_spectra_normed.sel(range=400, method='nearest').plot()
plt.show()


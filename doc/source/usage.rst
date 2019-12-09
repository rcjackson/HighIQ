============
Using HighIQ
============


Input and Output
----------------
HighIQ uses NetCDF files as its inputs. Currently, the I/O module supports NetCDF
files that conform to Atmospheric Radiation Measurement standards. Therefore, in
order to place data into a format supported by HighIQ, a data ingest must be
able to place the data in xarray datasets that contain 3D variables that have
dimensions of (*time*, *nsamples*, *complex*). These dimensions are defined as:

    * *time* = the number of samples in time
    * *nsamples* = the number of given samples for a ray
    * *complex* = index 0 is real component of ACF, 1 is imaginary component.

The *complex* dimension will always be of length 2. This array will store the ACF
for the radar or lidar signal. In addition, HighIQ also expects a background ACF
for determining the noise floor of the instrument. This background ACF array will
have the same dimensions as the ACF array.

After the acf array has been created (or loaded using xarray.open_dataset), the
next step is to create the Doppler spectra using :func:`highiq.calc.get_psd`::

    $ my_ds = highiq.calc.get_psd(my_ds)

Read the documentation on :func:`highiq.calc.get_psd` in order to learn how to
customize parameters such as the size of the range gate, sampling frequency, and
number of points to include in the FFT. This function will add Doppler spectra
variables 'power_spectra_density' which contains the power spectral density in
:math:`dB\ m\ s^{-1}` and 'power_spectra_density_normed' which is normalized
such that the integral under the curve is 1. The power spectra density are defined
at *nfft* points between the negative and positive Nyquist velocities of the
radar or lidar.

However, while *nfft* can be the optimal number of points to include in
the FFT, for a robust calculation of moments, interpolation of the Doppler spectra
are required. Therefore, In addition, HighIQ will interpolate the Doppler spectra
generated over the original *nfft* point space in order to provide smoother Doppler
spectra more suitable for the calculation of the moments of the spectra.

After these Doppler spectra are created, the calculation of the lidar moments is as
easy as::

    $ my_moments = highiq.calc.get_lidar_moments(my_ds)

Optimizing processing
---------------------
For both the processing of the Doppler spectra and moments, HighIQ will only store
a portion of the dataset in the GPU's memory due to limitations. However, if you
have a high amount of GPU memory, you may be able to optimize the processing
by increasing the *


After the Doppler spectra are generated

Plotting
--------
In order to make plots of the output Doppler spectra.

3D Visualization of spectra is easily done using the `Atmospheric Community Toolkit
(ACT) <https://anl-digr.github.io/ACT>`_ One is encouraged to read the documentation
and examples from ACT in order to learn how to create custom visualizations.


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

Plotting
--------
Visualization of spectra is easily done using the `Atmospheric Community Toolkit
(ACT) <https://anl-digr.github.io/ACT>`_ One is encouraged to read the documentation
and examples from ACT in order to learn how to create custom visualizations.
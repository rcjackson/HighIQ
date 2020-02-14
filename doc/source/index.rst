.. HighIQ documentation master file, created by
   sphinx-quickstart on Fri Dec  6 13:14:02 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to HighIQ's documentation!
==================================

This is a package for processing radar and lidar autocorrelation function (ACF)
data using NVIDIA GPUs. This was built with the intent to be ran on CUDA-compatible computers,
such as Argonne National Laboratory's `Waggle <https://wa8.gl>`_ nodes.
Therefore, much radar processing can be done live with the ability to create an initial processed
product (i.e. NOAA's Level 2 or ARM's b1) on any computer with a CUDA-compatible device.

This toolkit is based around the `Atmospheric Community Toolkit <https://arm-doe.github.io/ACT/>`_
which uses `xarray <https://xarray.pydata.org>`_ as its data model. ACT is a toolkit that supports
analysis and visualization of the data that is produced by HighIQ. The built in I/O module
supports `Atmospheric Radiation Measurement <https://www.arm.gov>`_'s Doppler lidar
autocorrelation function files (*dlacf.a0*). This product derives intensity, doppler velocity, spectral
width, skewness, and kurtosis from raw lidar signals.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   source/auto_examples/index
   contributing
   API/index.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

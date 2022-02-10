"""
===========
highiq.calc
===========

.. currentmodule:: highiq.calc

This module contains the methods that do the core calculations of HighIQ.

.. autosummary::
    :toctree: generated/

    get_lidar_moments
    get_psd
    calc_num_peaks

"""
from .moments import get_lidar_moments
from .spectra import get_psd, calc_num_peaks

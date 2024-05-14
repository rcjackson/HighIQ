import numpy as np
import warnings
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    import numpy as cp
    CUPY_AVAILABLE = False
    warnings.warn("Jax not installed...reverting to Numpy!", Warning)
import xarray as xr


def _gpu_calc_power(psd, dV, block_size=200, normed=True):
    shp = psd.shape
    power = np.zeros((shp[0], shp[1]))
    if len(shp) == 3:
        gpu_array = cp.array(psd, dtype=cp.float32)
        if normed:
            gpu_array = gpu_array * dV
        gpu_array = cp.sum(gpu_array, axis=2)
        if CUPY_AVAILABLE:
            power = gpu_array.asnumpy()
        else:
            power = gpu_array
    else:
        gpu_array = cp.array(psd.values, dtype=cp.float32)
        if normed:
            gpu_array = gpu_array * dV
        gpu_array = cp.sum(gpu_array, axis=1)
        if CUPY_AVAILABLE:
            power = gpu_array.asnumpy()
        else:
            power = gpu_array
    return power


def _gpu_calc_velocity(psd, power, vel_bins, dV):
    shp = psd.shape
    gpu_array = cp.array(psd, dtype=cp.float32)
    power_array = cp.array(power, dtype=cp.float32)
    vel_bins_tiled = cp.tile(vel_bins, (shp[0], shp[1], 1))
    gpu_array = 1 / power_array * cp.sum(gpu_array * vel_bins_tiled, axis=2)
    if CUPY_AVAILABLE:
        velocity = gpu_array.asnumpy()
    else:
        velocity = gpu_array
    return velocity


def _gpu_calc_velocity_dumb(psd, vel_bins):
    dV = np.diff(vel_bins)[0]
    vel_min = vel_bins.min()
    gpu_array = cp.array(psd)
    gpu_array = cp.argmax(gpu_array, axis=2)
    gpu_array = vel_min + gpu_array.astype(cp.float32) * dV
    if CUPY_AVAILABLE:
        velocity = gpu_array.asnumpy()
    else:
        velocity = gpu_array
    return velocity


def _gpu_calc_spectral_width(psd, power, vel_bins, velocity, dV):
    shp = psd.shape
    times = shp[0]
    specwidth = cp.zeros((shp[0], shp[1]))

    gpu_array = cp.array(psd.values, dtype=cp.float32)
    power_array = cp.array(power, dtype=cp.float32)

    velocity_array = cp.transpose(np.tile(velocity, (shp[2], 1, 1)), [1, 2, 0])
    vel_bins_tiled = cp.tile(vel_bins, (times, shp[1], 1))
    gpu_array = cp.sqrt(1 / power_array * cp.sum(
                             (vel_bins_tiled - velocity_array)**2 * gpu_array, axis=2))
    if CUPY_AVAILABLE:
        specwidth = gpu_array.asnumpy()
    else:
        specwidth = gpu_array
    return specwidth


def _gpu_calc_skewness(psd, power, vel_bins, velocity, spec_width, dV):
    shp = psd.shape
    times = shp[0]
    gpu_array = cp.array(psd.values, dtype=cp.float32)
    power_array = cp.array(power, dtype=cp.float32)
    spec_width_array = cp.array(spec_width, dtype=cp.float32)
    power_array *= spec_width_array**3

    velocity_array = cp.transpose(np.tile(velocity, (shp[2], 1, 1)), [1, 2, 0])
    vel_bins_tiled = cp.tile(vel_bins, (times, shp[1], 1))
    gpu_array = 1 / power_array * cp.sum(
        (vel_bins_tiled - velocity_array)**3 * gpu_array, axis=2)
    if CUPY_AVAILABLE:
        skewness = gpu_array.asnumpy()
    else:
        skewness = gpu_array
    return skewness


def _gpu_calc_kurtosis(psd, power, vel_bins, velocity, spec_width, dV):
    shp = psd.shape
    kurtosis = np.zeros((shp[0], shp[1]))
    gpu_array = cp.array(psd.values, dtype=cp.float32)
    power_array = cp.array(power, dtype=cp.float32)
    spec_width_array = cp.array(spec_width, dtype=cp.float32)
    power_array *= spec_width_array**4
    velocity_array = cp.transpose(cp.tile(velocity, (shp[2], 1, 1)), [1, 2, 0])
    vel_bins_tiled = cp.tile(vel_bins, (shp[0], shp[1], 1))
    gpu_array = 1 / power_array * cp.sum(
        (vel_bins_tiled - velocity_array)**4 * gpu_array, axis=2)
    if CUPY_AVAILABLE:
        kurtosis = gpu_array.asnumpy()
    else:
        kurtosis = gpu_array
    return kurtosis


def get_lidar_moments(spectra, intensity_thresh=0, block_size_ratio=1.0, which_moments=None):
    """
    This function will retrieve the lidar moments of the Doppler spectra.

    Parameters
    ----------
    spectra: ACT Dataset
        The dataset containing the processed Doppler spectral density functions.
    intensity_thresh: float
        The minimum signal to noise ratio to use as an initial mask of noise.
    block_size_ratio: float
        This value is used to determine how much data the GPU will process in one loop. If your
        GPU has more memory, you may be able to optimize processing by raising this number. In
        addition, if you encounter out of memory errors, try lowering this number, ensuring that
        it is a positive floating point number.
    which_moments: list or None
        This tells HighIQ which moments should be processed. If this list is None, then the
        signal to noise ratio, doppler velocity, spectral width, skewness,
        and kurtosis will be calculated.

    Returns
    -------
    spectra: ACT Dataset
        The database with the Doppler lidar moments.
    """
    if 'power_spectral_density' not in spectra.variables.keys():
        raise ValueError("You must calculate the power spectra before calculating moments!")
    if which_moments is None:
        which_moments = ['intensity', 'doppler_velocity', 'spectral_width',
                         'skewness', 'kurtosis']
    else:
        which_moments = [x.lower() for x in which_moments]

    if not block_size_ratio > 0:
        raise ValueError("block_size_ratio must be a positive floating point number!")

    dV = np.diff(spectra['vel_bins'])[0]
    linear_psd = spectra['power_spectral_density'] - 1
    linear_psd_0filled = linear_psd.fillna(0).values
    power = _gpu_calc_power(
        linear_psd_0filled, dV)
    velocity = _gpu_calc_velocity(linear_psd_0filled, power, spectra['vel_bins'].values, dV)

    if 'intensity' in which_moments:
        power_with_noise = dV * spectra['power_spectral_density'].sum(dim='vel_bins')
        spectra['intensity'] = power_with_noise / (dV * len(spectra['vel_bins']))
        spectra['intensity'].attrs['long_name'] = "Signal to Noise Ratio + 1"
        spectra['intensity'].attrs['units'] = ""
        spectra.attrs['intensity_mask'] = "%f" % intensity_thresh

    if 'doppler_velocity' in which_moments:
        velocity_dumb = _gpu_calc_velocity_dumb(
            linear_psd_0filled, spectra['vel_bins'].values)
        spectra['doppler_velocity_max_peak'] = xr.DataArray(
            velocity_dumb, dims=('time', 'range'))
        spectra['doppler_velocity_max_peak'].attrs['long_name'] = \
            "Doppler velocity derived using location of highest " \
            "peak in spectra."
        spectra['doppler_velocity_max_peak'].attrs["units"] = "m s-1"
        spectra['doppler_velocity'] = xr.DataArray(
            velocity, dims=('time', 'range'))
        spectra['doppler_velocity'].attrs['long_name'] = \
            "Doppler velocity using first moment"
        spectra['doppler_velocity'].attrs['units'] = "m s-1"
        spectra['doppler_velocity_max_peak'] = \
            spectra['doppler_velocity_max_peak'].where(spectra.intensity > intensity_thresh)
        spectra['doppler_velocity'] = spectra['doppler_velocity'].where(
            spectra.intensity > intensity_thresh)

    if 'spectral_width' in which_moments or 'kurtosis' in which_moments or 'skewness' in which_moments:
        spectral_width = _gpu_calc_spectral_width(
            linear_psd, power, spectra['vel_bins'].values,
            velocity, dV)

    if 'spectral_width' in which_moments:
        spectra['spectral_width'] = xr.DataArray(
            spectral_width, dims=('time', 'range'))
        spectra['spectral_width'].attrs["long_name"] = "Spectral width"
        spectra['spectral_width'].attrs["units"] = "m s-1"
        if 'intensity' in which_moments:
            spectra['spectral_width'] = spectra['spectral_width'].where(spectra.intensity > intensity_thresh)

    if 'skewness' in which_moments:
        skewness = _gpu_calc_skewness(
            linear_psd, power, spectra['vel_bins'].values, velocity, spectral_width, dV)
        spectra['skewness'] = xr.DataArray(skewness, dims=('time', 'range'))
        if 'intensity' in which_moments:
            spectra['skewness'] = spectra['skewness'].where(spectra.intensity > intensity_thresh)
        spectra['skewness'].attrs["long_name"] = "Skewness"
        spectra['skewness'].attrs["units"] = "m^3 s^-3"

    if 'kurtosis' in which_moments:
        kurtosis = _gpu_calc_kurtosis(
            linear_psd, power, spectra['vel_bins'].values, velocity, spectral_width, dV)
        spectra['kurtosis'] = xr.DataArray(kurtosis, dims=('time', 'range'))
        if 'intensity' in which_moments:
            spectra['kurtosis'] = spectra['kurtosis'].where(spectra.intensity > intensity_thresh)
        spectra['kurtosis'].attrs["long_name"] = "Kurtosis"
        spectra['kurtosis'].attrs["units"] = "m^4 s^-4"

    spectra['range'].attrs['long_name'] = "Range"
    spectra['range'].attrs['units'] = 'm'
    spectra['vel_bins'].attrs['long_name'] = "Doppler velocity"
    spectra['vel_bins'].attrs['units'] = 'm s-1'
    return spectra

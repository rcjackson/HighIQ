import numpy as np
import warnings
try:
    import cupy as cp
    cp.zeros(1)  # quick allocation test
    xp = cp
    print("Using CuPy (GPU)")
    from cupyx.scipy.signal import find_peaks
except Exception:
    import numpy as np
    xp = np
    print("Using NumPy (CPU)")
    from scipy.signal import find_peaks

import xarray as xr


def _gpu_calc_power(psd, dV, block_size=200, normed=True):
    shp = psd.shape
    power = xp.zeros((shp[0], shp[1]))
    if len(shp) == 3:
        gpu_array = xp.array(psd, dtype=xp.float32)
        if normed:
            gpu_array = gpu_array * dV
        gpu_array = xp.sum(gpu_array, axis=2)
        power = gpu_array.get() if hasattr(gpu_array, "get") else gpu_array
        
    else:
        gpu_array = xp.array(psd.values, dtype=xp.float32)
        if normed:
            gpu_array = gpu_array * dV
        gpu_array = xp.sum(gpu_array, axis=1)
        power = gpu_array.get() if hasattr(gpu_array, "get") else gpu_array
        
    return power


def _gpu_calc_velocity(psd, power, vel_bins, dV):
    shp = psd.shape
    gpu_array = xp.array(psd, dtype=xp.float32)
    power_array = xp.array(power, dtype=xp.float32)
    vel_bins_tiled = xp.tile(vel_bins, (shp[0], shp[1], 1))
    gpu_array = 1 / power_array * xp.sum(gpu_array * vel_bins_tiled, axis=2)
    velocity = gpu_array
    return velocity


def _gpu_calc_velocity_dumb(psd, vel_bins):
    dV = np.diff(vel_bins)[0]
    vel_min = vel_bins.min()
    gpu_array = xp.array(psd)
    gpu_array = xp.argmax(gpu_array, axis=2)
    gpu_array = vel_min + gpu_array.astype(xp.float32) * dV
    velocity = gpu_array
    return velocity


def _gpu_calc_spectral_width(psd, power, vel_bins, velocity, dV):
    shp = psd.shape
    times = shp[0]
    specwidth = xp.zeros((shp[0], shp[1]))
    gpu_array = xp.array(psd.values, dtype=xp.float32)
    power_array = xp.array(power, dtype=xp.float32)
    velocity_array = xp.transpose(xp.tile(velocity, (shp[2], 1, 1)), [1, 2, 0])
    vel_bins_tiled = xp.tile(vel_bins, (times, shp[1], 1))
    gpu_array = xp.sqrt(
        1
        / power_array
        * xp.sum((vel_bins_tiled - velocity_array) ** 2 * gpu_array, axis=2)
    )
    specwidth = gpu_array
    return specwidth


def _gpu_calc_skewness(psd, power, vel_bins, velocity, spec_width, dV):
    shp = psd.shape
    times = shp[0]
    gpu_array = xp.array(psd.values, dtype=xp.float32)
    power_array = xp.array(power, dtype=xp.float32)
    spec_width_array = xp.array(spec_width, dtype=xp.float32)
    power_array *= spec_width_array**3

    velocity_array = xp.transpose(xp.tile(velocity, (shp[2], 1, 1)), [1, 2, 0])
    vel_bins_tiled = xp.tile(vel_bins, (times, shp[1], 1))
    gpu_array = (
        1
        / power_array
        * xp.sum((vel_bins_tiled - velocity_array) ** 3 * gpu_array, axis=2)
    )
    skewness = gpu_array
    return skewness


def _gpu_calc_kurtosis(psd, power, vel_bins, velocity, spec_width, dV):
    shp = psd.shape
    kurtosis = np.zeros((shp[0], shp[1]))
    gpu_array = xp.array(psd.values, dtype=xp.float32)
    power_array = xp.array(power, dtype=xp.float32)
    spec_width_array = xp.array(spec_width, dtype=xp.float32)
    power_array *= spec_width_array**4
    velocity_array = xp.transpose(xp.tile(velocity, (shp[2], 1, 1)), [1, 2, 0])
    vel_bins_tiled = xp.tile(vel_bins, (shp[0], shp[1], 1))
    gpu_array = (
        1
        / power_array
        * xp.sum((vel_bins_tiled - velocity_array) ** 4 * gpu_array, axis=2)
    )
    kurtosis = gpu_array
    return kurtosis


def get_lidar_moments(
    spectra, intensity_thresh=0, block_size_ratio=1.0, which_moments=None
):
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
    if "power_spectral_density" not in spectra.variables.keys():
        raise ValueError(
            "You must calculate the power spectra before calculating moments!"
        )
    if which_moments is None:
        which_moments = [
            "intensity",
            "radial_velocity",
            "spectral_width",
            "skewness",
            "kurtosis",
        ]
    else:
        which_moments = [x.lower() for x in which_moments]

    if not block_size_ratio > 0:
        raise ValueError("block_size_ratio must be a positive floating point number!")

    dV = np.diff(spectra["vel_bins"])[0]
    linear_psd = spectra["power_spectral_density"] - 1
    linear_psd_0filled = linear_psd.fillna(0).values
    power = _gpu_calc_power(linear_psd_0filled, dV)
    velocity = _gpu_calc_velocity(
        linear_psd_0filled, power, spectra["vel_bins"].values, dV
    )

    if "intensity" in which_moments:
        power_with_noise = dV * spectra["power_spectral_density"].sum(dim="vel_bins")
        spectra["intensity"] = power_with_noise / (dV * len(spectra["vel_bins"]))
        spectra["intensity"].attrs["long_name"] = "Signal to Noise Ratio + 1"
        spectra["intensity"].attrs["units"] = ""
        spectra.attrs["intensity_mask"] = "%f" % intensity_thresh

    if "radial_velocity" in which_moments:
        velocity_dumb = _gpu_calc_velocity_dumb(
            linear_psd_0filled, spectra["vel_bins"].values
        )
        velocity_dumb = velocity_dumb.get() if hasattr(velocity_dumb, "get") else velocity_dumb
        spectra["radial_velocity_max_peak"] = xr.DataArray(
            velocity_dumb, dims=("time", "range")
        )
        spectra["radial_velocity_max_peak"].attrs[
            "long_name"
        ] = "Doppler velocity derived using location of highest peak in spectra."
        spectra["radial_velocity_max_peak"].attrs["units"] = "m s-1"
        velocity = velocity.get() if hasattr(velocity, "get") else velocity
        spectra["radial_velocity"] = xr.DataArray(velocity, dims=("time", "range"))
        spectra["radial_velocity"].attrs[
            "long_name"
        ] = "Doppler velocity using first moment"
        spectra["radial_velocity"].attrs["units"] = "m s-1"
        spectra["radial_velocity_max_peak"] = spectra["radial_velocity_max_peak"].where(
            spectra.intensity > intensity_thresh
        )
        spectra["radial_velocity"] = spectra["radial_velocity"].where(
            spectra.intensity > intensity_thresh
        )

    if (
        "spectral_width" in which_moments
        or "kurtosis" in which_moments
        or "skewness" in which_moments
    ):
        spectral_width = _gpu_calc_spectral_width(
            linear_psd, power, spectra["vel_bins"].values, velocity, dV
        )

    if "spectral_width" in which_moments:
        spectral_width = spectral_width.get() if hasattr(spectral_width, "get") else spectral_width
        spectra["spectral_width"] = xr.DataArray(spectral_width, dims=("time", "range"))
        spectra["spectral_width"].attrs["long_name"] = "Spectral width"
        spectra["spectral_width"].attrs["units"] = "m s-1"
        if "intensity" in which_moments:
            spectra["spectral_width"] = spectra["spectral_width"].where(
                spectra.intensity > intensity_thresh
            )

    if "skewness" in which_moments:
        skewness = _gpu_calc_skewness(
            linear_psd, power, spectra["vel_bins"].values, velocity, spectral_width, dV
        )
        skewness = skewness.get() if hasattr(skewness, "get") else skewness
        spectra["skewness"] = xr.DataArray(
            skewness, dims=("time", "range"))
        if "intensity" in which_moments:
            spectra["skewness"] = spectra["skewness"].where(
                spectra.intensity > intensity_thresh
            )
        spectra["skewness"].attrs["long_name"] = "Skewness"
        spectra["skewness"].attrs["units"] = "m^3 s^-3"

    if "kurtosis" in which_moments:
        kurtosis = _gpu_calc_kurtosis(
            linear_psd, power, spectra["vel_bins"].values, velocity, spectral_width, dV
        )
        kurtosis = kurtosis.get() if hasattr(kurtosis, "get") else kurtosis
        spectra["kurtosis"] = xr.DataArray(
            kurtosis, dims=("time", "range"))
        if "intensity" in which_moments:
            spectra["kurtosis"] = spectra["kurtosis"].where(
                spectra.intensity > intensity_thresh
            )
        spectra["kurtosis"].attrs["long_name"] = "Kurtosis"
        spectra["kurtosis"].attrs["units"] = "m^4 s^-4"

    spectra["vel_bins"].attrs["long_name"] = "Doppler velocity"
    spectra["vel_bins"].attrs["units"] = "m s-1"

    return spectra

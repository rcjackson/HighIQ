import numpy as np
import warnings
try:
    import jax.numpy as jnp
except ImportError:
    import numpy as jnp
    warnings.warn("Jax not installed...reverting to Numpy!", Warning)
import xarray as xr

from scipy.signal import find_peaks
from scipy.ndimage import convolve1d
from pint import UnitRegistry


def _fast_expand(complex_array, factor, num_per_block=200):
    shp = complex_array.shape
    expanded_array = np.zeros((shp[0], shp[1], shp[2] * factor))
    weights = np.tile(np.arange(0, factor) / factor, (shp[0], shp[1], 1))
    for i in range(shp[2]):
        gpu_array = jnp.zeros(complex_array[:, :, i], dtype=jnp.float32)
        if i < shp[2] - 1:
            gpu_array2 = jnp.zeros(complex_array[:, :, i + 1], dtype=jnp.float32)
            diff_array = gpu_array2 - gpu_array
        else:
            diff_array = jnp.zeros((shp[0], shp[1]))

        rep_array = jnp.transpose(
            jnp.tile(gpu_array, (factor, 1, 1)), [1, 2, 0])
        diff_array = jnp.transpose(
            jnp.tile(diff_array, (factor, 1, 1)), [1, 2, 0])
        temp_array = diff_array * weights + rep_array
        expanded_array[:, :, factor * i:factor * (i + 1)] = np.array(temp_array)
    return expanded_array


def get_psd(spectra, gate_resolution=60., wavelength=None, fs=None, nfft=1024, time_window=None,
            acf_name='acf', acf_bkg_name='acf_bkg', block_size_ratio=1.0,
            smooth_window=5):
    """
    This function will get the power spectral density from the autocorrelation function.

    Parameters
    ----------
    spectra: ACT Dataset
        The dataset containing the autocorrelation function.
    gate_resolution: float
        The gate resolution to derive the spectra for.
    wavelength: float or None
        The wavelength (in m) of the radar. If None, HighIQ will attempt to load the
        wavelength from the wavelength attribute of the spectra dataset.
    fs: float or None
        The pulse repetition frequency of the radar in Hz. If None,
        HighIQ will try and extract this information from the
        ACT dataset. This will require that the dataset
        have an attribute called sample_rate that contains a string
        with the magnitude and units of the sample rate.
    nfft: int
        The number of points to include in the FFT.
    time_window: int
        The number of time points to include in the rolling window.
    acf_name: str
        The name of the autocorrelation function field.
    acf_bkg_name: str
        The name of the autocorrelation function of the background.
    block_size_ratio: float
        Increase this value to use more GPU memory for processing. Doing this can
        poentially optimize processing.
    smooth_window: int
        Apply running average to power spectra to remove small scale noise using
        this window size.

    Returns
    -------
    spectra: ACT Dataset
        The dataset containing the power spectral density functions.
    """

    Q_ = UnitRegistry().Quantity
    if fs is None:
        if "sample_rate" not in spectra.attrs:
            raise KeyError("If sample frequency is not specified, then ACT Dataset must contain a sample_rate attribute!")

        fs_pint = Q_(spectra.attrs["sample_rate"])
        fs_pint = fs_pint.to("Hz")
        fs = fs_pint.magnitude

    if wavelength is None:
        if "wavelength" not in spectra.attrs:
            raise KeyError("If wavelength is not specified, then the dataset must contain a wavelength attribute!")
        fs_pint = Q_(spectra.attrs["wavelength"])
        fs_pint = fs_pint.to("m")
        wavelength = fs_pint.magnitude

    if time_window is not None:
        spectra = spectra.resample(time='%ds' % int(time_window)).mean()
    else:
        spectra[acf_bkg_name] = xr.DataArray(np.ones(spectra[acf_name].shape),
                                             dims=spectra[acf_name].dims) * spectra[acf_bkg_name]

    num_gates = int(gate_resolution / 3)
    complex_coeff_in = spectra[acf_name].isel(complex=0).values + \
        spectra[acf_name].isel(complex=1).values * 1j

    complex_coeff = np.zeros(
        (complex_coeff_in.shape[0], int(complex_coeff_in.shape[1] / num_gates),
            complex_coeff_in.shape[2]), dtype=np.complex128)
    for i in range(complex_coeff.shape[1]):
        complex_coeff[:, i, :] = np.sum(complex_coeff_in[:, (num_gates * i):(num_gates * i + 1), :], axis=1)
    del complex_coeff_in
    freq = np.fft.fftfreq(nfft) * fs
    spectra.attrs['nyquist_velocity'] = "%f m s-1" % (wavelength / (4 * 1 / fs))
    spectra['freq_bins'] = xr.DataArray(freq, dims=['freq'])
    spectra['freq_bins'].attrs["long_name"] = "Doppler spectra bins in frequency units"
    spectra['freq_bins'].attrs["units"] = "Hz"
    vel_bins = spectra['freq_bins'] * (wavelength / 2)
    inds_sorted = np.argsort(vel_bins.values)
    vel_bins = vel_bins[inds_sorted]
    attrs_dict = {'long_name': 'Doppler velocity', 'units': 'm s-1'}
    spectra['vel_bins'] = xr.DataArray(vel_bins, dims=('vel_bins'), attrs=attrs_dict)
    spectra['freq_bins'] = spectra['freq_bins'][inds_sorted]

    complex_coeff_bkg_in = (spectra[acf_bkg_name].isel(complex=0).values + spectra[acf_bkg_name].isel(complex=1).values * 1j)

    complex_coeff_bkg = np.zeros(
        (complex_coeff_bkg_in.shape[0], int(complex_coeff_bkg_in.shape[1] / num_gates),
            complex_coeff_bkg_in.shape[2]), dtype=np.complex128)
    for i in range(complex_coeff.shape[1]):
        complex_coeff_bkg[:, i, :] = np.sum(complex_coeff_bkg_in[:, (num_gates * i):(num_gates * i + 1), :], axis=1)
    num_lags = complex_coeff_bkg_in.shape[2]
    if nfft < num_lags:
        raise RuntimeError("Number of points in FFT < number of lags in sample!")
    pad_after = int((nfft - num_lags))
    pad_before = 0
    pad_lengths = [(0, 0), (0, 0), (pad_before, pad_after)]
    frames = jnp.pad(complex_coeff, pad_lengths, mode='constant', constant_values=0)
    window = 1 / smooth_window * np.ones(smooth_window) 
    power = convolve1d(np.abs(jnp.fft.fft(frames)),
        axis=2, weights=window)
    attrs_dict = {'long_name': 'Range', 'units': 'm'}
    spectra['range'] = xr.DataArray(
        gate_resolution * np.arange(int(frames.shape[1])),
        dims=('range'), attrs=attrs_dict)
    spectra['power'] = xr.DataArray(
        power[:, :, inds_sorted], dims=(('time', 'range', 'vel_bins')))
    frames = jnp.pad(complex_coeff_bkg, pad_lengths, mode='constant', constant_values=0)
    power = convolve1d(np.abs(jnp.fft.fft(frames)),
        axis=2, weights=window)

    spectra['power_bkg'] = xr.DataArray(
        power[:, :, inds_sorted], dims=('time', 'range', 'vel_bins'))

    # Subtract background noise
    spectra['power_spectral_density'] = spectra['power'] / spectra['power_bkg']

    # Ground noise floor to 1
    spectra['power_spectral_density'] = spectra['power_spectral_density'] - spectra['power_spectral_density'].min(axis=2) + 1
    spectra['power_spectral_density'].attrs["long_name"] = "Power spectral density"
    spectra['power_spectral_density'].attrs["units"] = ''
    spectra['range'].attrs['long_name'] = "Range"
    spectra['range'].attrs['units'] = 'm'
    spectra['vel_bins'].attrs['long_name'] = "Doppler velocity"
    spectra['vel_bins'].attrs['units'] = 'm s-1'
    return spectra


def calc_num_peaks(my_spectra, **kwargs):
    """

    This function will calculate the number of peaks in the spectra.

    Parameters
    ----------
    my_spectra: ACT Dataset
        The dataset to calculate the number of peaks for.
    kwargs:
        Additional keyword arguments are passed into :func:`scipy.signal.find_peaks`.
        The default minimum height and width of the peak are set to 3 and 8 points
        respectively.

    Returns
    -------
    my_spectra: ACT Dataset
        The dataset with an 'npeaks' variable included that shows the number of peaks.
    """
    spectra = my_spectra['power_spectral_density']
    my_array = spectra.fillna(0).values
    shp = my_array.shape
    num_peaks = np.zeros((shp[0], shp[1]))
    peak_loc = np.nan * np.ones((shp[0], shp[1], 5))

    vel_bins = my_spectra['vel_bins'].values

    if 'prominence' not in kwargs.keys():
        prominence = 0.01
    else:
        prominence = kwargs.pop('prominence')

    if 'width' not in kwargs.keys():
        width = 8
    else:
        width = kwargs.pop('width')

    if 'height' not in kwargs.keys():
        height = 1.5
    else:
        height = kwargs.pop('height')

    for i in range(shp[0]):
        for j in range(shp[1]):
            peaks = find_peaks(my_array[i, j], prominence=prominence, width=width, height=height, **kwargs)[0]
            num_peaks[i, j] = len(peaks)
            for k in range(len(peaks)):
                if k > 4:
                    continue
                peak_loc[i, j, k] = vel_bins[peaks[k]]

    my_spectra['npeaks'] = xr.DataArray(num_peaks, dims=('time', 'range'))
    my_spectra['npeaks'].attrs['long_name'] = "Number of peaks in Doppler spectra"
    my_spectra['npeaks'].attrs['units'] = "1"
    my_spectra['peak_velocities'] = xr.DataArray(peak_loc, dims=('time', 'range', 'peak_no'))
    my_spectra['peak_velocities'].attrs['long_name'] = "Dopper velocity peaks"
    my_spectra['peak_velocities'].attrs['units'] = 'm s-1'
    return my_spectra

import numpy as np
import tensorflow as tf
import xarray as xr

from scipy.signal import hann, find_peaks
from scipy.ndimage import convolve1d
from pint import UnitRegistry


def _fast_expand(complex_array, factor, num_per_block=200):
    shp = complex_array.shape
    times = shp[0]
    expanded_array = np.zeros((shp[0], shp[1], shp[2] * factor))
    weights = np.tile(np.arange(0, factor) / factor, (shp[0], shp[1], 1))
    for l in range(shp[2]):
        gpu_array = tf.constant(complex_array[:, :, l], dtype=tf.float32)
        if l < shp[2] - 1:
            gpu_array2 = tf.constant(complex_array[:, :, l + 1], dtype=tf.float32)
            diff_array = gpu_array2 - gpu_array
        else:
            diff_array = tf.zeros((shp[0], shp[1]))

        rep_array = tf.transpose(
            np.tile(gpu_array, (factor, 1, 1)), [1, 2, 0])
        diff_array = tf.transpose(
            np.tile(diff_array, (factor, 1, 1)), [1, 2, 0])
        temp_array = diff_array * weights + rep_array
        expanded_array[:, :, factor * l:factor * (l + 1)] = temp_array.numpy()
    return expanded_array

def get_psd(spectra, gate_resolution=30., wavelength=None, fs=None, nfft=32,
            acf_name='acf', acf_bkg_name='acf_bkg', block_size_ratio=1.0):
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
    acf_name: str
        The name of the autocorrelation function field.
    acf_bkg_name: str
        The name of the autocorrelation function of the background.
    block_size_ratio: float
        Increase this value to use more GPU memory for processing. Doing this can
        poentially optimize processing.

    Returns
    -------
    spectra: ACT Dataset
        The dataset containing the power spectral density functions.
    """

    Q_ = UnitRegistry().Quantity
    if fs is None:
        if not "sample_rate" in spectra.attrs:
            raise KeyError("If sample frequency is not specified, then ACT Dataset must contain a sample_rate " +
                           "attribute!")

        fs_pint = Q_(spectra.attrs["sample_rate"])
        fs_pint = fs_pint.to("Hz")
        fs = fs_pint.magnitude


    if wavelength is None:
        if not "wavelength" in spectra.attrs:
            raise KeyError("If wavelength is not specified, then the dataset must contain a wavelength attribute!")
        fs_pint = Q_(spectra.attrs["wavelength"])
        fs_pint = fs_pint.to("m")
        wavelength = fs_pint.magnitude


    num_gates = int(gate_resolution / 3)
    complex_coeff = spectra[acf_name].isel(complex=0).values +\
                    spectra[acf_name].isel(complex=1).values * 1j
    complex_coeff = np.reshape(complex_coeff,
                               (complex_coeff.shape[0],
                                int(complex_coeff.shape[1] / (num_gates)),
                                int(complex_coeff.shape[2] * num_gates)))
    ntimes = complex_coeff.shape[0]
    frames = tf.signal.frame(complex_coeff, frame_length=int(nfft),
                             frame_step=16, pad_end=True)
    window = tf.signal.hann_window(32).numpy()
    multiples = (frames.shape[0], frames.shape[1], frames.shape[2], 1)
    window = np.tile(window, multiples)
    power = np.square(tf.math.abs(tf.signal.fft(frames * window)))
    power = power.mean(axis=2)
    freq = np.fft.fftfreq(nfft) * fs
    attrs_dict = {'long_name': 'Range', 'units': 'm'}
    spectra['range'] = xr.DataArray(
        gate_resolution * np.arange(int(frames.shape[1])),
        dims=('range'), attrs=attrs_dict)

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

    spectra['power'] = xr.DataArray(
        power[:, :, inds_sorted], dims=(('time', 'range', 'vel_bins')))

    complex_coeff = (spectra[acf_bkg_name].isel(complex=0).values +
                     spectra[acf_bkg_name].isel(complex=1).values * 1j)
    complex_coeff = np.reshape(complex_coeff,
                               (int(complex_coeff.shape[0] / num_gates),
                                int(complex_coeff.shape[1] * num_gates)))

    frames = tf.signal.frame(complex_coeff, frame_length=int(nfft),
                             frame_step=16, pad_end=True)
    window = tf.signal.hann_window(32).numpy()
    multiples = (frames.shape[0], frames.shape[1],  1)
    window = np.tile(window, multiples)
    power = np.square(tf.abs(tf.signal.fft(frames * window)))
    power = power.mean(axis=1)
    spectra['power_bkg'] = xr.DataArray(
        power[:, inds_sorted], dims=('range', 'vel_bins'))

    # Subtract background noise
    spectra['power_spectral_density'] = spectra['power'] - \
                                        np.tile(spectra['power_bkg'], (ntimes, 1, 1))
    spectra['power_spectral_density'] = spectra['power_spectral_density'].where(
        spectra['power_spectral_density'] > 0, 0)
    spectra['power_spectral_density'].attrs["long_name"] = "Power spectral density"
    dV = np.diff(spectra['vel_bins'])
    tot_power = np.sum(spectra['power_spectral_density'].values, axis=2) * dV[1]

    power_tiled = np.stack(
        [tot_power for x in range(spectra['power'].values.shape[2])], axis=2)
    spectra['power_spectra_normed'] = spectra['power_spectral_density'] / power_tiled / dV[1] * 100
    spectra['power_spectra_normed'].attrs["long_name"] = "p.d.f. of power spectra"
    spectra['power_spectra_normed'].attrs["units"] = "%"
    spectra['power_spectral_density'] = 10 * np.log10(spectra['power_spectral_density']) / dV[1]
    spectra['power_spectral_density'].attrs["units"] = 's dB-1 m-1 '

    # Smooth out power spectra
    interpolated_bins = np.linspace(
        spectra['vel_bins'].values[0], spectra['vel_bins'].values[-1], 256)
    spectra['vel_bin_interp'] = xr.DataArray(interpolated_bins, dims=('vel_bin_interp'))

    my_array = tf.nn.conv2d(
        np.expand_dims(_fast_expand(spectra['power_spectral_density'].values, 8), axis=3),
        np.ones((1, 16, 1, 1)) / 16, 1, 'SAME').numpy()[:, :, :, 0]
    spectra['power_spectral_density_interp'] = xr.DataArray(
        my_array, dims=('time', 'range', 'vel_bin_interp'))
    spectra['power_spectral_density_interp'].attrs['long_name'] = "Power spectral density"
    spectra['power_spectral_density_interp'].attrs["units"] = "s dB-1 m-1"
    my_array = tf.nn.conv2d(
        np.expand_dims(_fast_expand(spectra['power_spectra_normed'].values, 8), axis=3),
        np.ones((1, 16, 1, 1)) / 16, 1, 'SAME').numpy()[:, :, :, 0]
    spectra['power_spectra_normed_interp'] = xr.DataArray(
        my_array, dims=('time', 'range', 'vel_bin_interp'),)
    spectra['power_spectra_normed_interp'].attrs['long_name'] = "p.d.f of power spectra"
    spectra['power_spectra_normed_interp'].attrs["units"] = "%"

    spectra['range'].attrs['long_name'] = "Range"
    spectra['range'].attrs['units'] = 'm'
    spectra['vel_bins'].attrs['long_name'] = "Doppler velocity"
    spectra['vel_bins'].attrs['units'] = 'm s-1'
    spectra['vel_bin_interp'].attrs['long_name'] = "Doppler velocity"
    spectra['vel_bin_interp'].attrs["units"] = "m s-1"
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
    spectra = my_spectra['power_spectra_normed_interp']
    my_array = spectra.fillna(0).values
    shp = my_array.shape
    num_peaks = np.zeros((shp[0], shp[1]))

    if not 'height' in kwargs.keys():
        height = 3
    else:
        height = kwargs.pop('height')

    if not 'width' in kwargs.keys():
        width = 8
    else:
        width = kwargs.pop('height')

    for i in range(shp[0]):
        for j in range(shp[1]):
            num_peaks[i, j] = len(
                find_peaks(my_array[i,j], height=height, width=width, **kwargs)[0])
    my_spectra['npeaks'] = xr.DataArray(num_peaks, dims=('time', 'range'))
    my_spectra['npeaks'].attrs['long_name'] = "Number of peaks in Doppler spectra"
    my_spectra['npeaks'].attrs['units'] = "1"

    return my_spectra
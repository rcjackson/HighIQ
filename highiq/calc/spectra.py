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

def get_psd(spectra, gate_resolution=60., wavelength=None, fs=None, nfft=1024,
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
    complex_coeff_in = spectra[acf_name].isel(complex=0).values +\
                       spectra[acf_name].isel(complex=1).values * 1j
    num_lags = complex_coeff_in.shape[1]
    complex_coeff = np.zeros(
            (complex_coeff_in.shape[0], int(complex_coeff_in.shape[1] / num_gates),
                complex_coeff_in.shape[2]), dtype=np.complex128)
    for i in range(complex_coeff.shape[1]):
        complex_coeff[:, i, :] = np.sum(complex_coeff_in[
                :, (num_gates * i):(num_gates * i+1), :], axis=1)
    del complex_coeff_in
    
    ntimes = complex_coeff.shape[0]
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

    complex_coeff_bkg_in = (spectra[acf_bkg_name].isel(complex=0).values +
                        spectra[acf_bkg_name].isel(complex=1).values * 1j)

    complex_coeff_bkg = np.zeros(
            (int(complex_coeff_bkg_in.shape[0] / num_gates), complex_coeff_bkg_in.shape[1]),
            dtype=np.complex128)
    for i in range(complex_coeff_bkg.shape[0]):
        complex_coeff_bkg[i, :] = np.sum(complex_coeff_bkg_in[
                (num_gates * i):(num_gates * i+1), :], axis=0)

    frames = tf.signal.frame(complex_coeff,
                             frame_length=int(nfft),
                             frame_step=int(nfft), pad_end=True)
    multiples = (frames.shape[0], frames.shape[1], frames.shape[2], 1)
    power = np.abs(tf.signal.fft(frames).numpy())
    power = power.mean(axis=2)
    attrs_dict = {'long_name': 'Range', 'units': 'm'}
    spectra['range'] = xr.DataArray(
        gate_resolution * np.arange(int(frames.shape[1])),
        dims=('range'), attrs=attrs_dict)
    spectra['power'] = xr.DataArray(
        power[:, :, inds_sorted], dims=(('time', 'range', 'vel_bins')))
    frames = tf.signal.frame(complex_coeff_bkg, frame_length=int(nfft),
                             frame_step=int(nfft), pad_end=True)
    multiples = (frames.shape[0], frames.shape[1],  1)
    power = np.abs(tf.signal.fft(frames).numpy())
    power = power.mean(axis=1)
    spectra['power_bkg'] = xr.DataArray(
        power[:, inds_sorted], dims=('range', 'vel_bins'))

    # Subtract background noise
    spectra['power_spectral_density'] = spectra['power'] / \
                                        np.tile(spectra['power_bkg'], (ntimes, 1, 1))
    spectra['power_spectral_density'] = spectra['power_spectral_density'].where(
        spectra['power_spectral_density'] > 0, 0)
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

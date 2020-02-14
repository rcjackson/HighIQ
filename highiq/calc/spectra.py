import numpy as np
import cupy as cp
import xarray as xr

from scipy.signal import hann, find_peaks
from pint import UnitRegistry
from .gpu_methods import _fast_expand, _gpu_moving_average


def welchs_method(complex_coeff, fs=50e6, nfft=32, window_skip=16, num_per_block=200):
    """
    This technique is an implementation of Welch's method for processing a Doppler spectral
    density function from complex autocorrelation function data. Welch's method calculates
    the Doppler spectral density function by calculating the FFT over (potentially overlapping)
    windows and then taking the average of the magnitudes of each FFT to create the Doppler spectra.

    Parameters
    ----------
    complex_coeff: complex nD array
        An n-D array of complex floats representing the value of the autocorrelation
        function. The first dimension is the number of entries in time, and the second is the number
        of entries in the range dimension.
    fs:
        The pulse repetition frequency of the radar.
    nfft:
        The number of points to include in the FFT for calculating spectra. This must
        be an even number.
    window_skip:
        The number of points to go forward for each window. This cannot exceed the
        size of the FFT.
    num_per_block:
        The number of time periods to send to the GPU at one time. This is implemented
        because the GPU can only handle a limited number of data points at one time.

    Returns
    -------
    freq: 1D array
        The frequencies corresponding to each value in the spectra.
    power: nD array
        The power spectral densities for each frequency.
    """

    if nfft % 2 == 1:
        raise ValueError("The number of points in the FFT must be even!")

    if window_skip > nfft:
        raise ValueError("window_skip cannot be greater than nfft!")

    times = complex_coeff.shape[0]
    num_points = complex_coeff.shape[-1]
    window = hann(nfft)
    if len(complex_coeff.shape) == 3:
        my_fft = np.zeros((complex_coeff.shape[0], complex_coeff.shape[1], nfft))
        for k in range(0, times, num_per_block):
            j = 0
            the_max = min([k + num_per_block, times])
            gpu_complex = cp.array(complex_coeff[k:the_max, :, :])
            windowt = cp.tile(window, (gpu_complex.shape[0], gpu_complex.shape[1], 1))
            temp_fft = cp.zeros((gpu_complex.shape[0], gpu_complex.shape[1], nfft))
            for i in range(0, num_points, window_skip):
                if i + nfft > num_points:
                    start = num_points - nfft
                    temp_fft += cp.square(
                        cp.abs(cp.fft.fft(gpu_complex[:, :, start:] * windowt, axis=-1)))
                else:
                    temp_fft += cp.square(
                        cp.abs(cp.fft.fft(gpu_complex[:, :, i:i + nfft] * windowt, axis=-1)))
                j += 1
            temp_fft = temp_fft / j
            my_fft[k:the_max] = temp_fft.get()
    elif len(complex_coeff.shape) == 2:
        j = 0
        gpu_complex = cp.array(complex_coeff)
        windowt = cp.tile(window, ((gpu_complex.shape[0], 1)))
        temp_fft = cp.zeros((gpu_complex.shape[0], nfft))
        for i in range(0, num_points, window_skip):
            if i + nfft > num_points:
                start = num_points - nfft
                temp_fft += cp.square(
                    cp.abs(cp.fft.fft(gpu_complex[:, start:] * windowt, axis=-1)))
            else:
                temp_fft += cp.square(
                    cp.abs(cp.fft.fft(gpu_complex[:, i:i + nfft] * windowt, axis=-1)))
            j += 1
        temp_fft = temp_fft / j
        my_fft = temp_fft.get()
    power = my_fft
    freq = np.fft.fftfreq(nfft) * fs

    return freq, np.array(power)


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
    complex_coeff = spectra[acf_name].sel(complex=1).values +\
                    spectra[acf_name].sel(complex=2).values * 1j
    complex_coeff = np.reshape(
        complex_coeff, (complex_coeff.shape[0],
                        int(complex_coeff.shape[1] / num_gates),
                        int(complex_coeff.shape[2] * num_gates)))
    freq, power = welchs_method(
        complex_coeff, fs=fs, nfft=nfft, num_per_block=int(block_size_ratio*200))
    attrs_dict = {'long_name': 'Range', 'units': 'm'}
    spectra['range'] = xr.DataArray(
        gate_resolution * np.arange(int(complex_coeff.shape[1])),
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

    complex_coeff = (spectra[acf_bkg_name].sel(complex=1).values +
                     spectra[acf_bkg_name].sel(complex=2).values * 1j)
    complex_coeff = np.reshape(
        complex_coeff, (complex_coeff.shape[0],
                        int(complex_coeff.shape[1] / num_gates),
                        int(complex_coeff.shape[2] * num_gates)))
    freq, power = welchs_method(
        complex_coeff, fs=50e6, nfft=32, num_per_block=int(200*block_size_ratio))
    spectra['power_bkg'] = xr.DataArray(
        power[:, :, inds_sorted], dims=(('time', 'range', 'vel_bins')))

    # Subtract background noise
    spectra['power_spectral_density'] = spectra['power'] - spectra['power_bkg']
    spectra['power_spectral_density'] = spectra['power_spectral_density'].where(
        spectra['power_spectral_density'] > 0, 0)
    spectra['power_spectral_density'].attrs["long_name"] = "Power spectral density"
    tot_power = np.sum(spectra['power_spectral_density'].values, axis=2)
    dV = np.diff(spectra['vel_bins'])
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
    my_array = _gpu_moving_average(
        _fast_expand(spectra['power_spectral_density'].values, 8))
    spectra['power_spectral_density_interp'] = xr.DataArray(
        my_array, dims=('time', 'range', 'vel_bin_interp'))
    spectra['power_spectral_density_interp'].attrs['long_name'] = "Power spectral density"
    spectra['power_spectral_density_interp'].attrs["units"] = "s dB-1 m-1"
    my_array = _gpu_moving_average(
        _fast_expand(spectra['power_spectra_normed'].values, 8))
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
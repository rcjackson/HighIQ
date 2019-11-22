import numpy as np
import cupy as cp
import xarray as xr
from scipy.signal import hann


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


def get_psd(spectra, gate_resolution=30, wavelength=1.548e-6, fs=50e6, nfft=32,
            acf_name='acf', acf_bkg_name='acf_bkg'):
    num_gates = int(gate_resolution / 3)
    complex_coeff = spectra[acf_name].sel(complex=1).values +\
                    spectra[acf_name].sel(complex=2).values * 1j
    complex_coeff = np.reshape(
        complex_coeff, (complex_coeff.shape[0],
                        int(complex_coeff.shape[1] / num_gates),
                        int(complex_coeff.shape[2] * num_gates)))
    freq, power = welchs_method(complex_coeff, fs=fs, nfft=nfft)
    spectra['range'] = xr.DataArray(
        gate_resolution * np.arange(int(complex_coeff.shape[1])),
        dims=('range'))
    spectra['range'].attrs['long_name'] = "Range"
    spectra['range'].attrs['units'] = "m"
    spectra.attrs['nyquist_velocity'] = 1548e-9 / (4 * 1 / 50e6)
    spectra.attrs['nyquist_velocity_units'] = "m s-1"

    spectra['freq_bins'] = xr.DataArray(freq, dims=['freq'])
    spectra['freq_bins'].long_name = "Doppler spectra bins in frequency units"
    spectra['freq_bins'].units = "s-1"
    vel_bins = spectra['freq_bins'] * (wavelength / 2)
    inds_sorted = np.argsort(vel_bins.values)
    vel_bins = vel_bins[inds_sorted]
    spectra['vel_bins'] = xr.DataArray(vel_bins, dims=('vel_bins'))
    spectra['vel_bins'].attrs['long_name'] = "Doppler spectra velocity bins"
    spectra['vel_bins'].units = "m s-1"
    spectra['freq_bins'] = spectra['freq_bins'][inds_sorted]
    spectra['power'] = xr.DataArray(
        power[:, :, inds_sorted], dims=(('time', 'range', 'vel_bins')))

    complex_coeff = (spectra['acf_bkg'].sel(complex=1).values +
                     spectra['acf_bkg'].sel(complex=2).values * 1j)
    complex_coeff = np.reshape(complex_coeff,
        (int(complex_coeff.shape[0] / num_gates),
         int(complex_coeff.shape[1] * num_gates)))
    freq, power = welchs_method(complex_coeff, fs=50e6, nfft=32)
    spectra['power_bkg'] = xr.DataArray(power[:, inds_sorted], dims=(('range', 'vel_bins')))

    # Subtract background noise
    spectra['power_spectral_density'] = spectra['power'] - spectra['power_bkg']
    spectra['power_spectral_density'] = spectra['power_spectral_density'].where(
        spectra['power_spectral_density'] > 0, 0)
    tot_power = np.sum(spectra['power_spectral_density'].values, axis=2)
    dV = np.diff(spectra['vel_bins'])
    power_tiled = np.stack(
        [tot_power for x in range(spectra['power'].values.shape[2])], axis=2)
    spectra['power_spectra_normed'] = spectra['power_spectral_density'] / power_tiled / dV[1] * 100
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
    my_array = _gpu_moving_average(
        _fast_expand(spectra['power_spectra_normed'].values, 8))
    spectra['power_spectra_normed_interp'] = xr.DataArray(
        my_array, dims=('time', 'range', 'vel_bin_interp'),)
    return spectra

def calc_num_peaks(my_spectra):
    spectra = my_spectra['power_spectra_normed_interp']
    my_array = spectra.fillna(0).values
    shp = my_array.shape
    num_peaks = np.zeros((shp[0], shp[1]))
    for i in range(shp[0]):
        for j in range(shp[1]):
            num_peaks[i, j] = len(find_peaks(my_array[i,j], height=3, width=8)[0])
    my_spectra['npeaks'] = xr.DataArray(num_peaks, dims=('time', 'range'))
    return my_spectra
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

from pint import UnitRegistry


def _fast_expand(complex_array, factor, num_per_block=200):
    shp = complex_array.shape
    expanded_array = np.zeros((shp[0], shp[1], shp[2] * factor))
    weights = np.tile(np.arange(0, factor) / factor, (shp[0], shp[1], 1))
    for i in range(shp[2]):
        gpu_array = xp.zeros(complex_array[:, :, i], dtype=xp.float32)
        if i < shp[2] - 1:
            gpu_array2 = xp.zeros(complex_array[:, :, i + 1], dtype=xp.float32)
            diff_array = gpu_array2 - gpu_array
        else:
            diff_array = xp.zeros((shp[0], shp[1]))

        rep_array = xp.transpose(xp.tile(gpu_array, (factor, 1, 1)), [1, 2, 0])
        diff_array = xp.transpose(xp.tile(diff_array, (factor, 1, 1)), [1, 2, 0])
        temp_array = diff_array * weights + rep_array
        expanded_array[:, :, factor * i : factor * (i + 1)] = np.array(temp_array)
    return expanded_array


def get_psd(
    spectra,
    gate_resolution=60.0,
    wavelength=None,
    fs=None,
    nfft=1024,
    time_window=10,
    acf_name="acf",
    acf_bkg_name="acf_bkg",
    nsamples=4000,
    hamming_window=True,
):
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
    nsamples: int
        Number of samples in ACF
    hamming_window: bool
        Whether to apply a Hamming window to the data before performing the FFT.

    Returns
    -------
    spectra: ACT Dataset
        The dataset containing the power spectral density functions.
    """

    Q_ = UnitRegistry().Quantity
    if fs is None:
        if "sample_rate" not in spectra.attrs:
            raise KeyError(
                "If sample frequency is not specified, then ACT Dataset must contain a sample_rate attribute!"
            )

        fs_pint = Q_(spectra.attrs["sample_rate"])
        fs_pint = fs_pint.to("Hz")
        fs = fs_pint.magnitude

    if wavelength is None:
        if "wavelength" not in spectra.attrs:
            raise KeyError(
                "If wavelength is not specified, then the dataset must contain a wavelength attribute!"
            )
        fs_pint = Q_(spectra.attrs["wavelength"])
        fs_pint = fs_pint.to("m")
        wavelength = fs_pint.magnitude

    spectra_out = spectra.resample(time="%ds" % time_window).mean()
    acf = spectra_out[acf_name]
    num_times = acf.shape[0]
    num_gates = int(gate_resolution / 3)
    complex_coeff_in = acf.isel(complex=0).values + acf.isel(complex=1).values * 1j
    complex_coeff_in = xp.array(complex_coeff_in, dtype=xp.complex128)
    complex_coeff = xp.zeros(
        (
            num_times,
            int(complex_coeff_in.shape[1] / num_gates),
            complex_coeff_in.shape[2],
        ),
        dtype=xp.complex128,
    )
    for i in range(complex_coeff.shape[1]):
        complex_coeff[:, i, :] = xp.mean(
            complex_coeff_in[:, (num_gates * i) : (num_gates * (i + 1)), :], axis=1
        )
    complex_coeff_in = None
    complex_coeff = complex_coeff / xp.tile(
        complex_coeff[:, :, 0], (complex_coeff.shape[2], 1, 1)
    ).transpose(1, 2, 0)

    if hamming_window:
        # Only use right half of the Hamming window
        hamming_win = xp.hamming(complex_coeff.shape[2] * 2)[complex_coeff.shape[2] :]
        hamming_win = xp.tile(
            hamming_win, (complex_coeff.shape[0], complex_coeff.shape[1], 1)
        )
        complex_coeff = complex_coeff * hamming_win

    freq = np.fft.fftshift(np.fft.fftfreq(nfft) * fs)
    spectra_out.attrs["nyquist_velocity"] = "%f m s-1" % (wavelength / (4 * 1 / fs))
    spectra_out["freq_bins"] = xr.DataArray(freq, dims=["freq"])
    spectra_out["freq_bins"].attrs[
        "long_name"
    ] = "Doppler spectra bins in frequency units"
    spectra_out["freq_bins"].attrs["units"] = "Hz"
    vel_bins = spectra_out["freq_bins"] * (wavelength / 2)
    inds_sorted = np.argsort(vel_bins.values)
    vel_bins = vel_bins[inds_sorted]
    attrs_dict = {"long_name": "Doppler velocity", "units": "m s-1"}
    spectra_out["vel_bins"] = xr.DataArray(
        vel_bins, dims=("vel_bins"), attrs=attrs_dict
    )
    spectra_out["freq_bins"] = spectra_out["freq_bins"][inds_sorted]

    complex_coeff_bkg_in = (
        spectra_out[acf_bkg_name].isel(complex=0).values
        + spectra_out[acf_bkg_name].isel(complex=1).values * 1j
    )
    complex_coeff_bkg_in = xp.array(complex_coeff_bkg_in, dtype=xp.complex128)
    complex_coeff_bkg = xp.zeros(
        (
            num_times,
            int(complex_coeff_bkg_in.shape[1] / num_gates),
            complex_coeff_bkg_in.shape[2],
        ),
        dtype=xp.complex128,
    )
    for i in range(complex_coeff.shape[1]):
        complex_coeff_bkg[:, i, :] = xp.mean(
            complex_coeff_bkg_in[:, (num_gates * i) : (num_gates * (i + 1)), :], axis=1
        )
    complex_coeff_bkg = complex_coeff_bkg / xp.tile(
        complex_coeff_bkg[:, :, 0], (complex_coeff_bkg.shape[2], 1, 1)
    ).transpose(1, 2, 0)
    if hamming_window:
        hamming_win = xp.hamming(complex_coeff.shape[2] * 2)[complex_coeff.shape[2] :]
        hamming_win = xp.tile(
            hamming_win, (complex_coeff_bkg.shape[0], complex_coeff_bkg.shape[1], 1)
        )
        complex_coeff_bkg = complex_coeff_bkg * hamming_win

    num_lags = complex_coeff_bkg_in.shape[2]
    if nfft < num_lags:
        raise RuntimeError("Number of points in FFT < number of lags in sample!")
    frames = xp.asarray(complex_coeff)
    arr = xp.fft.fft(frames, n=nfft)
    power = xp.abs(xp.fft.fftshift(arr, axes=2))
    power = power.get() if hasattr(power, "get") else power

    attrs_dict = {"long_name": "Range", "units": "m"}
    spectra_out["range"] = xr.DataArray(
        gate_resolution * np.arange(power.shape[1]),
        dims=("range"),
        attrs=attrs_dict,
    )
    frames = xp.asarray(complex_coeff_bkg)
    arr = xp.fft.fft(frames, n=nfft)
    power_bkg = xp.abs(xp.fft.fftshift(arr, axes=2))
    power_bkg = power_bkg.get() if hasattr(power_bkg, "get") else power_bkg
    # Subtract background noise
    spectra_out["power_spectral_density"] = (
        ["time", "range", "vel_bins"],
        power / power_bkg,
    )

    # Ground noise floor to 1
    spectra_out["power_spectral_density"].attrs["long_name"] = "Power spectral density"
    spectra_out["power_spectral_density"].attrs["units"] = ""
    spectra_out["range"].attrs["long_name"] = "Range"
    spectra_out["range"].attrs["units"] = "m"
    spectra_out["vel_bins"].attrs["long_name"] = "Doppler velocity"
    spectra_out["vel_bins"].attrs["units"] = "m s-1"
    power = None
    power_bkg = None
    spectra_out = spectra_out.drop(["acf", "acf_bkg"])
    return spectra_out


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
    spectra = my_spectra["power_spectral_density"]
    my_array = xp.array(spectra.fillna(0).values)
    shp = my_array.shape
    num_peaks = xp.zeros((shp[0], shp[1]))
    peak_loc = xp.nan * xp.ones((shp[0], shp[1], 5))

    vel_bins = my_spectra["vel_bins"].values

    if "prominence" not in kwargs.keys():
        prominence = 0.01
    else:
        prominence = kwargs.pop("prominence")

    if "width" not in kwargs.keys():
        width = 8
    else:
        width = kwargs.pop("width")

    if "height" not in kwargs.keys():
        height = 1.5
    else:
        height = kwargs.pop("height")

    for i in range(shp[0]):
        for j in range(shp[1]):
            peaks = find_peaks(
                my_array[i, j],
                prominence=prominence,
                width=width,
                height=height,
                **kwargs,
            )[0]
            peaks = peaks.get() if hasattr(peaks, "get") else peaks
            num_peaks[i, j] = len(peaks)
            for k in range(len(peaks)):
                if k > 4:
                    continue
                peak_loc[i, j, k] = vel_bins[peaks[k]]
    num_peaks = num_peaks.get() if hasattr(num_peaks, "get") else num_peaks
    peak_loc = peak_loc.get() if hasattr(peak_loc, "get") else peak_loc
    my_spectra["npeaks"] = xr.DataArray(num_peaks, dims=("time", "range"))
    my_spectra["npeaks"].attrs["long_name"] = "Number of peaks in Doppler spectra"
    my_spectra["npeaks"].attrs["units"] = "1"
    my_spectra["peak_velocities"] = xr.DataArray(
        peak_loc, dims=("time", "range", "peak_no")
    )
    my_spectra["peak_velocities"].attrs["long_name"] = "Dopper velocity peaks"
    my_spectra["peak_velocities"].attrs["units"] = "m s-1"
    return my_spectra

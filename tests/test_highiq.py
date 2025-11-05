import highiq
import numpy as np


def test_io():
    my_ds = highiq.io.load_arm_netcdf(highiq.testing.TEST_FILE)
    assert "acf" in my_ds.variables.keys()
    assert "acf_bkg" in my_ds.variables.keys()
    my_ds.close()


def test_spectra():
    my_ds = highiq.io.load_arm_netcdf(highiq.testing.TEST_FILE)
    my_spectra = highiq.calc.get_psd(my_ds)
    assert "power_spectral_density" in my_spectra.variables.keys()
    psd = my_spectra["power_spectral_density"].sel(range=400, method="nearest")
    vel_bins = my_spectra["vel_bins"]
    dV = vel_bins[1] - vel_bins[0]
    np.testing.assert_almost_equal(
        psd.values.sum() * dV.values, 39.644146692612104, decimal=1
    )
    my_ds.close()
    my_spectra.close()


def test_moments():
    my_ds = highiq.io.load_arm_netcdf(highiq.testing.TEST_FILE)
    my_spectra = highiq.calc.get_psd(my_ds)
    my_moments = highiq.calc.get_lidar_moments(my_spectra)
    intensity = my_moments["intensity"].values
    velocity = my_moments["radial_velocity"].values
    assert np.nanmin(intensity) > -1.0
    assert np.nanmin(velocity) < -2.0
    my_ds.close()
    my_spectra.close()


def test_peaks():
    my_ds = highiq.io.load_arm_netcdf(highiq.testing.TEST_FILE)
    my_spectra = highiq.calc.get_psd(my_ds, gate_resolution=200.0)
    my_peaks = highiq.calc.calc_num_peaks(
        my_spectra, height=1.5, width=8, prominence=0.5
    )
    my_peaks = highiq.calc.get_lidar_moments(my_peaks)
    my_peaks["npeaks"] = my_peaks["npeaks"].where(my_peaks.intensity > 0.5)
    num_peaks = my_peaks["npeaks"].values
    assert np.nanmax(num_peaks) == 1
    my_ds.close()
    my_spectra.close()
    my_peaks.close()

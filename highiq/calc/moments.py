import numpy as np
import cupy as cp
import xarray as xr


def gpu_calc_power(psd, dV, block_size=200, normed=True):
    shp = psd.shape
    times = shp[0]
    power = np.zeros((shp[0], shp[1]))
    if len(shp) == 3:
        for k in range(0, times, block_size):
            the_max = min([k + block_size, times])
            gpu_array = cp.array(psd[k:the_max, :, :])
            if normed:
                gpu_array = 10 ** (gpu_array / 10. * dV)
            gpu_array = cp.sum(gpu_array, axis=2)
            power[k:the_max] = gpu_array.get()
    else:
        gpu_array = cp.array(psd)
        if normed:
            gpu_array = 10 ** (gpu_array / 10. * dV)
        gpu_array = cp.sum(gpu_array, axis=1)
        power = gpu_array.get()

    return power

def gpu_calc_velocity(psd, power, vel_bins, dV, block_size=100):
    shp = psd.shape
    times = shp[0]
    velocity = np.zeros((shp[0], shp[1]))
    print(psd.max())
    print(power.max())
    for k in range(0, times, block_size):
        the_max = min([k + block_size, times])
        gpu_array = cp.array(psd[k:the_max, :, :], dtype=float64)
        power_array = cp.array(power[k:the_max, :], dtype=float64)
        vel_bins_tiled = cp.tile(vel_bins, (the_max - k, shp[1], 1))
        gpu_array = 10 ** (gpu_array / 10. * dV)
        gpu_array = 1 / power_array * cp.sum(gpu_array * vel_bins_tiled, axis=2)
        velocity[k:the_max, :] = gpu_array.get()
    return velocity

def gpu_calc_velocity_dumb(psd, vel_bins, block_size=500):
    shp = psd.shape
    times = shp[0]
    velocity = np.zeros((shp[0], shp[1]))
    # Get dV and vel_min
    dV = np.diff(vel_bins)[0]
    vel_min = vel_bins.min()
    for k in range(0, times, block_size):
        the_max = min([k + block_size, times])
        gpu_array = cp.array(psd[k:the_max, :, :])
        gpu_array = cp.argmax(gpu_array, axis=2)
        gpu_array = vel_min + gpu_array * dV
        velocity[k:the_max, :] = gpu_array.get()
    return velocity

def gpu_calc_spectral_width(psd, power, vel_bins, velocity, dV, block_size=100):
    shp = psd.shape
    times = shp[0]
    specwidth = np.zeros((shp[0], shp[1]))
    for k in range(0, times, block_size):
        the_max = min([k+block_size, times])
        gpu_array = cp.array(psd[k:the_max, :, :], dtype=float64)
        power_array = cp.array(power[k:the_max, :], dtype=float64)
        velocity_array = cp.array(velocity[k:the_max, :])
        velocity_array = cp.transpose(cp.tile(velocity_array, (shp[2], 1, 1)), [1, 2, 0])
        vel_bins_tiled = cp.tile(vel_bins, (the_max-k, shp[1], 1))
        gpu_array = 10**(gpu_array/10.*dV)
        gpu_array = np1/power_array*cp.sum((vel_bins_tiled - velocity_array)**2 * gpu_array, axis=2)
        specwidth[k:the_max, :] = gpu_array.get()
    return specwidth

def gpu_snr(power, noise, block_size=500):
    shp = psd.shape
    times = shp[0]
    snr = np.zeros_like(power)
    #or k in range(0, times, block_size):
    #   the_max = min([k+block_size, times])
    gpu_power = cp.array(power)
    gpu_noise = cp.tile(cp.array(noise), (times, 1))
    gpu_power = 10*cp.log10(gpu_power/gpu_noise)
    snr = gpu_power.get()
    return snr
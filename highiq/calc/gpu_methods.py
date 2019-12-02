import numpy as np
import cupy as cp

def _fast_expand(complex_array, factor, num_per_block=200):
    shp = complex_array.shape
    times = shp[0]
    expanded_array = np.zeros((shp[0], shp[1], shp[2] * factor))
    for k in range(0, times, num_per_block):
        the_max = min([k + num_per_block, times])
        weights = cp.tile(np.arange(0, factor) / factor, (the_max - k, shp[1], 1))
        for l in range(shp[2]):
            gpu_array = cp.array(complex_array[k:the_max, :, l])
            if l < shp[2] - 1:
                gpu_array2 = cp.array(complex_array[k:the_max, :, l + 1])
                diff_array = gpu_array2 - gpu_array
            else:
                diff_array = cp.zeros((the_max - k, shp[1]))

            rep_array = cp.transpose(
                cp.tile(gpu_array, (factor, 1, 1)), [1, 2, 0])
            diff_array = cp.transpose(
                cp.tile(diff_array, (factor, 1, 1)), [1, 2, 0])
            temp_array = diff_array * weights + rep_array
            expanded_array[k:the_max, :, factor * l:factor * (l + 1)] = temp_array.get()
    return expanded_array


def _gpu_moving_average(arr, window=8, num_per_block=200):
    shp = arr.shape
    times = shp[0]
    for k in range(0, times, num_per_block):
        the_max = min([k + num_per_block, times])
        gpu_arr = cp.array(arr[k:the_max, :, :])
        gpu_arr2 = cp.zeros_like(gpu_arr)
        for i in range(shp[2]):
            the_min2 = max([0, i - int(window / 2)])
            the_max2 = min([i + int(window / 2), shp[2]])
            gpu_arr2[:, :, i] = cp.mean(gpu_arr[:, :, the_min2:the_max2], axis=2)
        arr[k:the_max, ::] = gpu_arr2.get()
        del gpu_arr, gpu_arr2
    return arr
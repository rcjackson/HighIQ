import os
import struct
import pandas as pd
import xarray as xr

from datetime import datetime

import numpy as np

global_attrs = {
    "number_of_lags": {"cor.M1": 7, "hou.M1": 7, "oli.M1": 7, "sgp.C1": 20},
    "number_of_samples_per_lag": {
        "cor.M1": 3200,
        "hou.M1": 3200,
        "oli.M1": 1920,
        "sgp.C1": 4000,
    },
}


def load_arm_netcdf(arm_file, **kwargs):
    """

    This loads netCDF data that are in the Atmospheric Radiation Measurement standard netCDF format.

    Parameters
    ----------
    arm_file: str
        The path to the dataset to load.

    Additional keyword arguments are passed into :func`xr.open_dataset`

    Returns
    -------
    ds: Xarray Dataset
        Returns the xarray dataset that contains the autocorrelation functions.
    """

    ds = xr.open_dataset(arm_file, **kwargs)
    if "time" not in ds["acf"].dims:
        ds["acf"] = ds["acf"].expand_dims("time")
    if "time" not in ds["acf_bkg"].dims:
        ds["acf_bkg"] = ds["acf_bkg"].expand_dims(dim={"time": len(ds.time)}, axis=0)
    return ds


def read_00_data(
    file_name, home_point, site="sgp.C1", sample_start=None, sample_end=None, **kwargs
):
    """
    This reads a raw StreamLine Doppler Lidar .00 level file.

    Parameters
    ----------
    file_name: str
        Name of .raw file to read
    home_point: str
        Name of home point file path.
    sample_start: int
        Start of time period to load (beam index)
    sample_end: int
        End of time periods to load (beam index)
    kwargs

    Returns
    -------
    ds: ACT Dataset
        Returns the ACT dataset (xarray dataset) that contains the autocorrelation functions.
    """
    with open(file_name, mode="rb") as f:
        home_point_columns = [
            "start_time (yyyymmddhh)",
            "end_time (yyyymmddhh)",
            "Target_latitude",
            "Target_longitude",
            "target_altitude",
            "lidar_latitude",
            "lidar_longitude",
            "lidar_altitude",
            "lidar_home_point",
            "Descriptive comment",
        ]
        num_rows_to_skip = 0
        with open(home_point) as hf:
            for line in hf:
                if line[0] == "#":
                    num_rows_to_skip += 1

        hfile = pd.read_csv(
            home_point, skiprows=num_rows_to_skip, names=home_point_columns
        )
        background_vals = (
            global_attrs["number_of_lags"][site]
            * global_attrs["number_of_samples_per_lag"][site]
            * 2
        )
        nlags = global_attrs["number_of_lags"][site]
        num_samples = global_attrs["number_of_samples_per_lag"][site]
        background_bytes = 8 * background_vals
        beam_bytes = 24 + background_bytes
        dot_split = file_name.split(".")
        if len(dot_split) < 3:
            dot_split = dot_split[0].split("_")
        for x in dot_split:
            if len(x) == 8 and x.isnumeric():
                date_str = x
            if len(x) == 6 and x.isnumeric():
                time_str = x
            under_split = x.split("_")
            for x in under_split:
                print(x)
                if len(x) == 8 and x.isnumeric():
                    date_str = x
                if len(x) == 2 and x.isnumeric():
                    time_str = x + "0000"

        base_time = datetime.strptime("%s.%s" % (date_str, time_str), "%Y%m%d.%H%M%S")
        midnight = datetime(
            year=base_time.year,
            month=base_time.month,
            day=base_time.day,
            hour=0,
            minute=0,
            second=0,
        ).timestamp()
        base_time_int = int(base_time.strftime("%Y%m%d%H%M%S"))
        for i in range(1, len(hfile)):
            num = int(hfile["start_time (yyyymmddhh)"][i])
            end_num = int(hfile["end_time (yyyymmddhh)"][i])
            if end_num == -1:
                end_num = np.inf
            if base_time_int >= num and base_time_int < end_num:
                target_latitude = float(hfile["Target_latitude"][i])
                target_longitude = float(hfile["Target_longitude"][i])
                target_altitude = float(hfile["target_altitude"][i])
                lidar_latitude = float(hfile["lidar_latitude"][i])
                lidar_longitude = float(hfile["lidar_longitude"][i])
                lidar_home_point = float(hfile["lidar_home_point"][i])

        if site[:3] == "oli":
            read_background = False
            total_sample_bytes = os.path.getsize(file_name)
        else:
            read_background = True
            total_sample_bytes = os.path.getsize(file_name) - background_bytes

        if sample_start is None and sample_end is None:
            nsamples = int(total_sample_bytes / beam_bytes)
        else:
            maxnsamples = int(total_sample_bytes / beam_bytes)
            if sample_end > maxnsamples:
                sample_end = maxnsamples
            nsamples = sample_end - sample_start

        if read_background:
            stringFmt = "<"
            for i in range(background_vals):
                stringFmt += "d"
            my_data = f.read(background_bytes)
            acf_bkg = np.array(struct.unpack(stringFmt, my_data))
            acf_bkg = np.reshape(acf_bkg, (nlags, num_samples, 2))
            acf_bkg = np.transpose(acf_bkg, [1, 0, 2])

        azimuth = np.zeros(nsamples)
        elevation = np.zeros(nsamples)
        timestamp = np.zeros(nsamples)
        acf = np.zeros((nsamples, num_samples, nlags, 2))

        stringFmt = "<"
        for i in range(background_vals):
            stringFmt += "d"
        previous_time = 0
        spill_over = 0.0
        if sample_start is not None:
            num_bytes_to_skip = beam_bytes * sample_start
            f.seek(num_bytes_to_skip, 1)

        for i in range(nsamples):
            data = f.read(8)
            if len(data) < 8:
                break
            azimuth[i] = float(struct.unpack("<d", data)[0]) + lidar_home_point
            data = f.read(8)
            elevation[i] = float(struct.unpack("<d", data)[0])
            data = f.read(8)
            timestamp[i] = float(struct.unpack("<d", data)[0])

            seconds_decimal = timestamp[i] * 3600
            seconds = int(seconds_decimal)
            seconds_fraction = seconds_decimal - float(seconds)
            seconds_since_1970 = seconds + midnight + seconds_fraction
            timestamp[i] = seconds_since_1970
            if previous_time == 0:
                previous_time = seconds_decimal
            if previous_time - seconds_decimal > 86400:
                spill_over += 86400
            previous_time = seconds_decimal
            seconds_since_1970 += spill_over

            data = f.read(background_vals * 8)
            acf_temp = np.array(struct.unpack(stringFmt, data))
            acf_temp = np.reshape(acf_temp, (nlags, num_samples, 2))
            acf_temp = np.transpose(acf_temp, [1, 0, 2])
            acf[i] = acf_temp

        azimuth = xr.DataArray(azimuth.astype(np.float32), dims=("time"))
        azimuth.attrs["long_name"] = "Azimuth angle"
        azimuth.attrs["units"] = "Degree"
        elevation = xr.DataArray(elevation.astype(np.float32), dims=("time"))
        elevation.attrs["units"] = "Degree"
        elevation.attrs["long_name"] = "Elevation angle"
        timestamp = np.array([datetime.fromtimestamp(x) for x in timestamp])
        timestamp = xr.DataArray(timestamp, dims=("time"))

        acf_bkg = xr.DataArray(
            acf_bkg.astype(np.float32), dims=("nsamples", "nlags", "complex")
        )
        acf_bkg.attrs["long_name"] = "Autocorrelation function of background"
        acf = xr.DataArray(
            acf.astype(np.float32), dims=("time", "nsamples", "nlags", "complex")
        )
        acf.attrs["long_name"] = "Autocorrelation function"
        lidar_ds = xr.Dataset(
            {
                "acf": acf,
                "acf_bkg": acf_bkg,
                "aziumth": azimuth,
                "time": timestamp,
                "elevation": elevation,
            }
        )
        lidar_ds = lidar_ds.sortby("time")
        lidar_ds.attrs["dlon"] = lidar_longitude
        lidar_ds.attrs["dlat"] = lidar_latitude
        lidar_ds.attrs["target_latitude"] = target_latitude
        lidar_ds.attrs["target_longitude"] = target_longitude
        lidar_ds.attrs["target_altitude"] = target_altitude
        lidar_ds.attrs["home_point_azimuth"] = lidar_home_point
        lidar_ds.attrs["sample_rate"] = "50 MHz"
        lidar_ds.attrs["wavelength"] = "1548 nm"
    return lidar_ds

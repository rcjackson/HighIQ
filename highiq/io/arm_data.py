import act


def load_arm_netcdf(arm_file, **kwargs):
    """

    This loads netCDF data that are in the Atmospheric Radiation Measurement standard netCDF format.
    This is a wrapper around :func:`act.io.armfiles.read_netcdf`

    Parameters
    ----------
    arm_file: str
        The path to the dataset to load.

    Additional keyword arguments are passed into :func:`act.io.armfiles.read_netcdf`

    Returns
    -------
    ds: ACT Dataset
        Returns the ACT dataset (xarray dataset) that contains the autocorrelation functions.
    """
    return act.io.armfiles.read_netcdf(arm_file, **kwargs)

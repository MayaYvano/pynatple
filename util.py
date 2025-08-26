
import numpy 
import xarray
import harmonica
import struct
import pandas

from xrft.xrft import fft as _fft
from xrft.xrft import ifft as _ifft
from numpy.typing import NDArray

from pynatple.pronounce import Hyperparameter, Individual #type: ignore

def binary(
    input:Hyperparameter,
) -> Individual:
    """
    Convert hyperparameter to binary string representation.
    """
    return ''.join(format(int(genome), '08b') 
                    for genome 
                    in struct.pack('ii', *input))
    

def unbinary(
    input: Individual,
) -> Hyperparameter:
    """
    Convert binary string representation back to hyperparameter value.
    """
    if not isinstance(input, str):
        raise TypeError("Input to unbinary must be a binary string.")
    bytes_val = int(input, 2).to_bytes((len(input) + 7) // 8, byteorder='big')
    return struct.unpack('ii', bytes_val)


def eval(
        data:NDArray | xarray.DataArray,
        metric:str,
) -> float:
    
    """
    Score the evaluation metrics between the data.
    ----------
    Parameters:
    data : NDArray, xarray.DataArray
        Gravity (or magnetic [?]) data.
    metric : str
        The evaluation metric to be used. Can be one of the following:
        - 'rmse': root mean square error
        - 'mae': mean absolute error
        - 'mse': mean square error
        - 'L1': L1 norm
        - 'L2': L2 norm
        - 'r2': r-squared
    """

    if isinstance(data, xarray.DataArray):
         data = data.values

    try:
        if metric == 'rmse':
            value = float(numpy.sqrt(numpy.nanmean((data**2))))
        elif metric == 'mae':
            value = float(numpy.nanmean(numpy.abs(data)))
        elif metric == 'mse':
            value = float(numpy.nanmean((data**2)))
        elif metric == 'L1':
            value = float(numpy.linalg.norm(data, ord=1))
        elif metric == 'L2':
            value = float(numpy.linalg.norm(data, ord=2))
        elif metric == 'r2': # Im not sure, code below is suggested by copilot.
            # Calculate the r-squared value
            ss_res = numpy.nansum((data - numpy.nanmean(data))**2)
            ss_tot = numpy.nansum((data - numpy.nanmean(data))**2)
            value = float(1 - (ss_res / ss_tot))
        else:
            msg = f"Unknown metric: {metric}. Please use one of the following: 'rmse', 'mae', 'mse', 'L1', 'L2', 'r2'."
            raise ValueError(msg)
    
    except RuntimeWarning as e:
        print(f'RuntimeWarning: {e}')
        raise

    return value

    

def wavenumber(
        grid:xarray.DataArray,  
) -> xarray.DataArray:
        
    """
    Calculate the wavenumber (k) from a grid.

    ------------
    Parameters:
    grid : xarray.DataArray
        Gravity (or magnetic [?]) data.
    
    ----------
    Returns:
    xarray.DataArray
        Wavenumber (k) of the grid.
    """

    #Perform FFT on the grid
    gfft = _fft(grid)
    #Get the dimensions of the grid
    dims = gfft.dims
    #Get the frequency coordinates of the grid
    freq_easting, freq_northing = gfft.coords[dims[1]], gfft.coords[dims[0]]
    k_easting, k_northing = 2 * numpy.pi * freq_easting, 2 * numpy.pi * freq_northing

    #Calculate the wavenumber
    k = numpy.sqrt(k_easting**2 + k_northing**2)

    return k.T


def grid_sanity_check(
        grid:xarray.DataArray | xarray.Dataset,
        # fill_value:str | None = None,
) -> None:
    
    """
    Run the sanity check on the grid. Right now is used to check wether the grid has nan or not.
    If the grid has nan, raise an error.

    ------------
    Parameters:
    grid : xarray.DataArray, xarray.Dataset
        Gravity (or magnetic [?]) data.
    
    ----------
    raises:
    ValueError
        If the grid has nan in it.

    """

    if numpy.isnan(grid).any():

        # if fill_value == 'mean':
        #     grid = grid.fillna(grid.mean())

        #     return grid
        
        # elif fill_value == 'median':
        #     grid = grid.fillna(grid.median())

        #     return grid
            
        msg = "This grid contain nan value(s). Please get rid of it first."
        raise ValueError(msg)    



def check_gravity_inside_topography_region(
    dataset: xarray.Dataset,
    topography:xarray.DataArray,
) -> None:
    
    """
    Check that all gravity data is inside the region of the topography grid.
    Adopted from invert4geom package.

    """

    # # Make sure the dims is easting and northing
    # if topography.dims != ("easting", "northing"):
    #     msg = "The topography's dimensions should be easting and northing."
    #     raise ValueError(msg)
    
    # if dataset.dims != ("easting", "northing"):
    #     msg = "The dataset's dimensions should be easting and northing."
    #     raise ValueError(msg)    

    # Get the topography's region
    topo_easting, topo_northing = topography.easting.values, topography.northing.values
    w, e, s, n = (numpy.min(topo_easting), numpy.max(topo_easting), numpy.min(topo_northing), numpy.max(topo_northing))

    # Get the dataset's coords
    ds_easting, ds_northing = dataset.easting.values, dataset.northing.values


    ### Code block bellow is based on verde package.

    # Allocate temporary arrays to minimize memory allocation overhead
    out = numpy.empty_like(ds_easting, dtype=bool)
    tmp = tuple(numpy.empty_like(ds_easting, dtype=bool) for i in range(4))
    # Using the logical functions is a lot faster than & > < for some reason
    # Plus, this way avoids repeated allocation of intermediate arrays
    in_we = numpy.logical_and(
        numpy.greater_equal(ds_easting, w, out=tmp[0]),
        numpy.less_equal(ds_easting, e, out=tmp[1]),
        out=tmp[2],
    )
    in_ns = numpy.logical_and(
        numpy.greater_equal(ds_northing, s, out=tmp[0]),
        numpy.less_equal(ds_northing, n, out=tmp[1]),
        out=tmp[3],
    )
    inside = numpy.logical_and(in_we, in_ns, out=out)

    ### End of verde code block.


    if not inside.all():
        msg = (
            "Some gravity data are outside the region of the topography grid. "
            "This may result in unexpected behavior."
        )
        raise ValueError(msg)


def grids_to_prisms(
    surface: xarray.DataArray,
    reference: float | xarray.DataArray,
    density: float | int | xarray.DataArray,
    input_coord_names: tuple[str, str] = ("easting", "northing"),
) -> xarray.Dataset:
    
    """
    create a Harmonica layer of prisms with assigned densities.
    Borrorwed from invert4geom package. 

    Parameters
    ----------
    surface : xarray.DataArray
        data to use for prism surface
    reference : float | xarray.DataArray
        data or constant to use for prism reference, if value is below surface, prism
        will be inverted
    density : float | int | xarray.DataArray
        data or constant to use for prism densities, should be in the form of a density
        contrast across a surface (i.e. between air and rock).
    input_coord_names : tuple[str, str], optional
        names of the coordinates in the input dataarray, by default
        ("easting", "northing")
    Returns
    -------
    xarray.Dataset
       a prisms layer with assigned densities
    """

    # if density provided as a single number, use it for all prisms
    if isinstance(density, (float, int)):
        dens = density * numpy.ones_like(surface)
    # if density provided as a dataarray, map each density to the correct prisms
    elif isinstance(density, xarray.DataArray):
        dens = density
    else:
        msg = "invalid density type, should be a number or DataArray"
        raise ValueError(msg)

    # create layer of prisms based off input dataarrays
    prisms = harmonica.prism_layer(
        coordinates = (
            surface[input_coord_names[0]].values,
            surface[input_coord_names[1]].values,
        ),
        surface = surface,
        reference = reference,
        properties = {
            "density": dens,
        },
    )

    prisms["thickness"] = prisms.top - prisms.bottom

    # add zref as an attribute
    return prisms.assign_attrs(zref = reference)


def extract_data(
    data: xarray.DataArray, 
    target_line: xarray.DataArray,
) -> xarray.DataArray:
    """
    Get the data values along the line.
    dims = [dim for dim in control_line_list[24].dims] --> to get the dims' name.
    """
    
    east = xarray.DataArray(
                numpy.linspace(
                    target_line.easting.min(),
                    target_line.easting.max(),
                    target_line.size,
                ),
            )
    north = xarray.DataArray(
                numpy.linspace(
                    target_line.northing.min(),
                    target_line.northing.max(),
                    target_line.size,
                ),
            )

    line = data.interp(easting = east, northing = north, method = 'cubic')
    return line


def get_eval_points(
    evaluated_data: xarray.DataArray,
    control_data: xarray.DataArray | pandas.DataFrame,
    method:str = 'nearest',
    tolerance: float | None = None,
    factor: float | None = None,
) -> numpy.ndarray:
    
    if evaluated_data.dims != ('northing', 'easting'):
        evaluated_data = evaluated_data.rename({evaluated_data.dims[0]: 'northing', evaluated_data.dims[1]: 'easting'})
    if method not in ['nearest', 'linear', 'cubic']:
        msg1 = "Method must be one of 'nearest', 'linear', or 'cubic'."
        raise ValueError(msg1)
    
    if isinstance(control_data, pandas.DataFrame):
        cols = ['easting', 'northing', 'depth']
        if all(col in control_data.columns for col in cols) is False:
            msg2 = "'control_data' DataFrame needs all the following columns: {cols}."
            raise ValueError(msg2)    
        controler = control_data.copy().set_index(['northing', 'easting']).to_xarray().depth

    elif isinstance(control_data, xarray.DataArray):
        controler = control_data.copy()
        if controler.dims != ('northing', 'easting'):
            controler = controler.rename({controler.dims[0]: 'northing', controler.dims[1]: 'easting'})
    

    controler_stacked = controler.stack(points = ['northing', 'easting'])
    controler_nonull = controler_stacked[controler_stacked.notnull()]

    point_list = []
    for i in range(len(controler_nonull)):
        point = evaluated_data.sel(
            northing = controler_nonull.indexes['points'][i][0],
            easting = controler_nonull.indexes['points'][i][1],
            method = method,
            tolerance = tolerance,
        )
        point_list.append(point.values)

    
    if len(point_list) == 0:
        msg3 = "No evaluation points found. Check the method and tolerance."
        raise ValueError(msg3)
    
    if factor is not None:
        point_score = controler_nonull.values - (factor - numpy.array(point_list))
        return point_score
    
    else:
        return numpy.array(point_list)


def airy_heiskanen_isostasy(
    topography: xarray.DataArray,
    crust_thickness: float,
    crust_density: float,
    mantle_density: float,
    water_density: float = 1040.,
) -> xarray.DataArray:
    
    """
    Calculate vetical depth variation for idealized isostatic compensation model based on
    Airy-Heiskanen hypothesis.

    Parameters
    ----------
    topography : xarray.DataArray
        Topography data.
    crust_thickness : float
        Constant thickness of the crust in meters.
    crust_density : float
        Density of the crust in kg/m^3.
    mantle_density : float
        Density of the mantle in kg/m^3.
    water_density : float, optional
        Density of water in kg/m^3, by default 1040.

    Returns
    -------
    xarray.DataArray
    """

    iso = xarray.where(topography < 0,
                    (topography * ((crust_density - water_density)) / (mantle_density - crust_density)),
                    (topography * crust_density) / (mantle_density - crust_density))
    
    if crust_thickness > 0:
        crust_thickness *= -1
    
    return crust_thickness - iso


def pratt_hayford_isostasy(
    topography: xarray.DataArray,
    crust_thickness: float,
    crust_density: float,
    water_density: float = 1040.,
) -> xarray.DataArray:
    
    """
    Calculate lateral density variation for idealized isostatic compensation model based on
    Pratt-Hayford hypothesis.

    Parameters
    ----------
    topography : xarray.DataArray
        Topography data.
    crust_thickness : float
        Constant thickness of the crust in meters.
    crust_density : float
        Density of the crust in kg/m^3.
    water_density : float, optional
        Density of water in kg/m^3, by default 1040.

    Returns
    -------
    xarray.DataArray
    """

    if crust_thickness < 0:
        crust_thickness *= -1

    return xarray.where(topography < 0,
            ((crust_density * crust_thickness + water_density * topography) / (crust_thickness + topography)),
            ((crust_density * crust_thickness) / (crust_thickness + topography)))


def vertical_tectonic_stress(
    moho_gravity: xarray.DataArray,
    moho_isostatic: xarray.DataArray,
    crust_density: float,
    mantle_density: float = 3300.,
    gravity_acceleration: float = 9.81,
) -> xarray.DataArray:
    """ 
    Calculate the vertical tectonic stress based on the difference between
    the gravity anomaly at the Moho and the isostatic compensation.
    Equation based on Gao et al. [2019].

    Gao, S., She, Y., & Fu, G. (2016). A new method for computing the vertical tectonic stress 
    of the crust by use of hybrid gravity and GPS data. Chinese Journal of Geophysics, 59(6), 2006–2013. 
    https://doi.org/10.6038/cjg20160607

    Parameters
    ----------
    moho_gravity : xarray.DataArray
        Moho depth based on gravitational observation. Positive downward.
    moho_isostatic : xarray.DataArray
        Moho depth based on isostatic compensation model. Preferably, calculated
        based on Airy-Heiskanen models. Positive downward.
    crust_density : float
        Density of the crust in kg/m^3.
    mantle_density : float, optional
        Density of the mantle in kg/m^3, by default 3300.
    gravity_acceleration : float, optional
        Gravitational acceleration in m/s^2, by default 9.81.
    
    Returns
    -------
    xarray.DataArray
        Vertical tectonic stress in Pa.
    """

    return ((moho_gravity - moho_isostatic.values) * (mantle_density - crust_density) * gravity_acceleration)
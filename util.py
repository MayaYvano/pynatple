
import numpy 
import xarray
import harmonica
import struct

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
    control_data: xarray.DataArray,
    method:str = 'nearest',
    tolerance: float | None = None,
):
    
    if evaluated_data.dims != ('northing', 'easting'):
        evaluated_data = evaluated_data.rename({evaluated_data.dims[0]: 'northing', evaluated_data.dims[1]: 'easting'})
    if control_data.dims != ('northing', 'easting'):
        control_data = control_data.rename({control_data.dims[0]: 'northing', control_data.dims[1]: 'easting'})
    
    if method not in ['nearest', 'linear', 'cubic']:
        msg = "Method must be one of 'nearest', 'linear', or 'cubic'."
        raise ValueError(msg)
    
    eval_points = evaluated_data.sel(
        easting = control_data.easting, 
        northing = control_data.northing, 
        method = method,
        tolerance = tolerance,
    )
    
    if eval_points.size == 0:
        msg = "No evaluation points found. Check the method and tolerance."
        raise ValueError(msg)
    
    return eval_points
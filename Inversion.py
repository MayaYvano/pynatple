
import numpy
import xarray
import math
import logging
import copy

from xrft.xrft import fft as _fft
from xrft.xrft import ifft as _ifft
from numpy.typing import NDArray
from typing import Any, List, Tuple, Optional

from pynatple import util # type: ignore[self-owned module]
from pynatple.pronounce import Hyperparameter, ForwardFunc, FilterFunc, InverseFunc # type: ignore

# set the log procedure
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.basicConfig(format = '%(name)s - %(levelname)s -> %(message)s')

#Constants
Newton_universal_gravity_constants = 6.67430e-11  # m^3 kg^-1 s^-2


# NOT USED ANYMORE. BUT NOT BECAUSE IT DOESN'T WORK PERFECTLY.
def taylor_expansion(
        elev:xarray.DataArray,
        n:int,
        kgrid:xarray.DataArray | NDArray | None = None,
) -> NDArray:
    
    """
    Taylor expansion for the algorithm. 

    ------------
    Parameters:
    elev : xarray.DataArray
        Known or assumed depth of the causative interface. 
    n : int
        Number of desired degree.
    kgrid : xarray.DataArray, optional
        Wavenumber vector. If None, it will be calculated from the topographical grid (elev).
    
    --------------
    Returns:
    xarray.DataArray
        Second term of the Oldenburg's reformulation. 
        Used to give a detail variance of depth or gravitational effect of the causative interface.
        The value is in wavenumber domain (pre-ifft).
    """

    vars_fft = xarray.zeros_like(
        other = elev,
        dtype = numpy.complex128,
    ).values

    if kgrid is None:    
        try:
            kgrid = util.wavenumber(elev)
        
        except:
            msg = "Either elevation grid or kgrid (i.e. wavenumber vector) must be provided."
            raise ValueError(msg)


    for ni in range(1, n + 1):
        var_fft = ((kgrid)**(ni - 1) / math.factorial(ni)) * _fft(elev**ni) # type: ignore
        vars_fft += var_fft

    return vars_fft


def parker_forward(
        elev:xarray.DataArray,
        hyperparam:Hyperparameter,
        lower_boundary:Optional[xarray.DataArray] = None,
        **kwargs: Any,
) -> xarray.DataArray:
    """
    Forward modeling algorithm to determine the gravitational effect of a causative interface with Fourier transform.
    The calculation is based on the Parker's [1973] work. Right now only support homogenous density contrast.

    ------------
    Parameters:
    elev : xarray.DataArray
        Elevation grid of the interface.
    hyperparam : Hypermarameter
        The hyperparameter for the modeling (density contrast and reference depth).
        The first element is the density contrast (kg/m^3) of the causative interface.
        The second element is the reference depth (m) of the interface.
    kwargs : dict, optional
        Additional keyword arguments.
        max_iteration : int, optional
            Maximum degree for the Taylor expansion. Default is 10.
        grav_unit : str, optional
            Unit of the output. Default is 'mGal'. Other options are 'SI' (m^2/s^2), the rest is t.b.a. later.
    
    ------------
    Returns:
    xarray.DataArray
        Gravitational effect of the causative interface. 
        The value is in wavenumber domain (pre-ifft).
        The unit is mGal.

    """
    
    G = Newton_universal_gravity_constants
    density_contrast, reference_depth = hyperparam
    max_iteration = kwargs.get('max_iteration', 10)
    unit = kwargs.get('grav_unit', 'mGal')

    kgrid = util.wavenumber(elev)

    first = -2 * numpy.pi * G * numpy.exp(-1 * kgrid * reference_depth)
    expansion = numpy.zeros(
        shape = elev.shape,
        dtype = numpy.complex128,
    )

    if isinstance(density_contrast, float | int):
        for n in range(1, max_iteration + 1):
            var = ((kgrid.values)**(n - 1) / math.factorial(n)) * _fft(elev**n)
            if var.isnull().any():
                break
            expansion += var.values

        gfft = first * density_contrast * expansion
    
    elif isinstance(density_contrast, numpy.ndarray):
        if lower_boundary is None:
            lower_boundary = xarray.zeros_like(elev)
        for n in range(1, max_iteration + 1):
            var = ((kgrid.values)**(n - 1) / math.factorial(n)) * _fft(density_contrast * (elev**n - lower_boundary**n))
            if var.isnull().any():
                break
            expansion += var.values

        gfft = first * expansion

    else:
        raise TypeError("Unsupported type for density_contrast: {}".format(type(density_contrast)))

    result = _ifft(gfft).real

    result.name = 'gravity'
    if unit == 'mGal':
        result = result.assign_attrs(
            {'unit': 'mGal'},
        )
        return result * 1e5 # SI to mGal

    return result


def parker_oldenburg_inversion(
    data:xarray.DataArray,
    hyperparam:Hyperparameter,
    filter_func:Optional[FilterFunc] = None,
    lower_cutoff:Optional[float] = None,
    upper_cutoff:Optional[float] = None,
    **kwargs: Any,    
) -> xarray.DataArray:
    
    """
    Inversion algorithm to determine the depth of the causative interface from the gravity data.
    The algorithm is based on the Oldenburg's [1974] reformulation of Parker's [1973] algorithm.

    ------------
    Parameters:
    data : xarray.DataArray
       Gravity grid. 
    hyperparam : tuple
        The hyperparameter for the modeling (density contrast and reference depth).
        The first element is the density contrast (kg/m^3) of the causative interface.
        The second element is the reference depth (m) of the interface.
    kwargs : dict, optional
        Additional keyword arguments.
        max_iteration : int, optional
            Maximum degree for the Taylor expansion. Default is 10.
        grav_unit : str, optional
            Unit of the output. Default is 'mGal'. Other options are 'SI' (m^2/s^2), 
            the rest is t.b.a. later.
    
    ------------
    Returns:
    xarray.DataArray
        Gravitational effect of the causative interface. 
        The value is in wavenumber domain (pre-ifft).
        The unit is mGal.

    """
    
    G = Newton_universal_gravity_constants
    density_contrast, reference_depth = hyperparam
    max_order = kwargs.get('max_order', 10)
    grav_unit = kwargs.get('grav_unit', 'mGal')

    grav = copy.deepcopy(data)
    if grav_unit == 'mGal':
        grav *= 1e-5
    
    kgrid = util.wavenumber(grav).values
    first = (_fft(grav) * numpy.exp((kgrid) * (reference_depth))) / (2 * numpy.pi * G * density_contrast)
    expansion = numpy.zeros(
        shape = grav.shape,
        dtype = numpy.complex128,
    )

    elev = None
    for n in range(1, max_order + 1):

        if n == 1:
            elev = _ifft(first).real
        else:
            # Ensure elev is defined before use
            if elev is None:
                elev = _ifft(first).real
            var = ((kgrid)**(n - 1) / math.factorial(n)) * _fft((elev)**n)
            if var.isnull().any():
                break
            expansion += var.values

            # Apply the filter.
            if filter_func is not None:
                if upper_cutoff is None or lower_cutoff is None:
                    msg3 = (
                        "Filter function was assigned, but upper and lower cutoff wavenumbers "
                        "are not provided. Please provide them first."
                    )
                    raise ValueError(msg3)
                else:
                    filter_mask = filter_func(kgrid, upper_cutoff, lower_cutoff)
                    expansion *= filter_mask

    elev = -1 * _ifft(first + expansion).real
    elev.name = 'inverted depth'
    elev = elev.assign_attrs(
        {
            'density_contrast': density_contrast,
            'reference_depth': reference_depth,
            'unit': 'm',
        }
    )
    return elev


def oldenburg_filter(
    wavenum:xarray.DataArray | NDArray,
    upper_cutoff:float,
    lower_cutoff:float,
) -> NDArray:
    
    """
    Oldenburg filter.
    
    Parameters
    ----------
    wavenum : xarray.DataArray or NDArray
        Wavenumbers to filter.
    upper_cutoff : float
        Depth upper (shallow) cutoff for estimated souce in meter.
    lower_cutoff : float
        Depth lower (deep) cutoff for estimated souce in meter.
    
    Returns
    -------
    NDArray
        Filtered wavenumbers.
    """
    
    if isinstance(wavenum, xarray.DataArray):
        wavenum = wavenum.values
    
    spatial_freq = numpy.abs(wavenum / (2 * numpy.pi))
    SH = 1 / upper_cutoff
    WH = 1 / lower_cutoff
    
    return numpy.where(
        spatial_freq < WH,
        1.0,
        numpy.where(
            spatial_freq > SH,
            0.0,
            0.5 * (1 + numpy.cos((spatial_freq - 2 * numpy.pi * WH) / (2 * (SH - WH))))
        )
    )


def update_misfit_and_gravity(
    observed_data:xarray.DataArray,
    inverted_param:xarray.DataArray,
    hyperparam: Hyperparameter,
    iteration_number: int,
    forward_func:ForwardFunc,
    eval_method:Optional[str] = 'rmse',
) -> Tuple[xarray.Dataset, float]:
    
    """
    It do like what it was named.

    Parameters:
    observed_data : xarray.DataArray
       Original gravity grid. 
    inverted_param : xarray.DataArray
       Result from previous inversion.
    hyperparam : Tuple[float, float]
        The hyperparameter for the modeling (density contrast and reference depth).
    iteration_number : int
        number of iteration performed.
    forward_func : ForwardFunc
        Forward modeling operator to calculate the gravity effect of inveted result.
    eval_method : str, optional
        Method to score the evaluation of misfit. Default is 'rmse'.
    
    ------------
    Returns:
    xarray.Dataset
        Consist of calculated gravity grid, the causative grid, and its misft.
        Also attributed with iteration number, evaluation method, and evaluation score.
    float
        Evaluation score.
    """
    
    calculated_data = xarray.DataArray(
        forward_func(inverted_param, hyperparam).values,
        coords = observed_data.coords,
        dims = observed_data.dims,
    )

    misfit = observed_data.values - calculated_data.values
    score = util.eval(misfit, metric = eval_method)

    result = xarray.Dataset(
        data_vars = dict(
            gravity = ([observed_data.dims[0], observed_data.dims[1]], calculated_data.values),
            inverted_depth = ([observed_data.dims[0], observed_data.dims[1]], inverted_param.values),
            misfit = ([observed_data.dims[0], observed_data.dims[1]], misfit),
        ),
        coords = observed_data.coords,
        attrs = dict(
            iteration = iteration_number,
            evaluation_method = eval_method,
            evaluation_score = score,
        )
    )

    return result, score


def terminate(
    iteration_number: int,
    max_iterations: int,
    rmses: List[float],
    rmse_tolerance: float,
    delta_rmse: float,
    previous_delta_rmse: float,
    delta_rmse_tolerance: float,
    increase_limit: float,
) -> Tuple[bool, List[str]]:
    """
    check if the inversion should be terminated.
    Brute-adopted from invert4geom package.

    Parameters
    ----------
    iteration_number : int
        the iteration number, starting at 1 not 0
    max_iterations : int
        the maximum allowed iterations, inclusive and starting at 1
    rmses : float
        a list of each iteration's rmse
    rmse_tolerance : float
        the rmse criterion to end the inversion at
    delta_rmse : float
        the current iteration's delta rmse
    previous_delta_rmse : float
        the delta rmse of the previous iteration
    delta_rmse_tolerance : float
        the delta rmse criterion to end the inversion at
    increase_limit : float
        the set tolerance for decimal increase relative to the starting rmse

    Returns
    -------
    end : bool
        whether or not to end the inversion
    termination_reason : list[str]
        a list of termination reasons
    """
    end = False
    termination_reason = []

    rmse = rmses[-1]

    # ignore for first iteration
    if iteration_number == 1:
        pass
    else:
        if rmse > numpy.min(rmses) * (1 + increase_limit):
            logger.info(
                f"\nInversion terminated after {iteration_number} iterations because \n"
                f"RMSE = {rmse} was over {(1 + increase_limit) * 100}% greater than \n"
                f"minimum RMSE ({numpy.min(rmses)}) \n"
                "Change parameter 'increase_limit' if desired.",
            )
            end = True
            termination_reason.append("rmse increasing")

        if (delta_rmse <= delta_rmse_tolerance) & (previous_delta_rmse <= delta_rmse_tolerance):
            logger.info(
                f"\nInversion terminated after {iteration_number} iterations because \n"
                "there was no significant variation in the L2-norm over 2 iterations \n"
                "Change parameter 'delta_l2_norm_tolerance' if desired.",
            )

            end = True
            termination_reason.append("delta rmse tolerance")

        if rmse < rmse_tolerance:
            logger.info(
                f"\nInversion terminated after {iteration_number} iterations because \n"
                f"RMSE = {rmse} was less than its tolerance ({rmse_tolerance}) \n"
                "Change parameter 'rmse_tolerance' if desired.",
            )

            end = True
            termination_reason.append("rmse tolerance")

    if iteration_number >= max_iterations:
        logger.info(
            f"\nInversion terminated after {iteration_number} iterations because \n"
            f"maximum number of {max_iterations} iterations reached."
        )

        end = True
        termination_reason.append("max iterations")

    if "max iterations" in termination_reason:
        msg = (
            "Inversion terminated due to max_iterations limit. Consider increasing "
            "this limit."
        )
        logger.warning(msg)

    return end, termination_reason


def run_inversion(
    data:xarray.Dataset | xarray.DataArray,
    hyperparam:Hyperparameter,
    forward_func:ForwardFunc = parker_forward,
    inverse_func:InverseFunc = parker_oldenburg_inversion,
    filter_func:Optional[FilterFunc] = oldenburg_filter,
    upper_cutoff:Optional[float] = 7.5e4,
    lower_cutoff:Optional[float] = 1.5e5,
    **kwargs:Any,
) -> Tuple[xarray.Dataset, List[float], List[float]]:
    
    """
    Inversion algorithm to determine the depth of the causative interface from the gravity data.
    The algorithm is based on the Oldenburg's [1974] reformulation of Parker's [1973] algorithm.

    ------------
    Parameters:
    data : xarray.Dataset
        Dataset containing the gravity data and the topographical grid.
        The dataset must contain the following variables:
            - 'gravity': gravity data (xarray.DataArray)
    hyperparam : tuple
        The hyperparameter for the modeling (density contrast and reference depth).
        The first element is the density contrast (kg/m^3) of the causative interface.
        The second element is the reference depth (m) of the interface.
    forward_func : callable, optional
        Forward modeling function. Default is built-in 'parker_forward'.
    inverse_func : callable, optional
        Inversion function. Default is built-in 'parker_oldenburg_inversion'.
    filter_func : callable, optional
        Filter function to apply on the inverted depth. Default is None.
    upper_cutoff : float, optional
        Upper cutoff wavenumber for the filter function. Must be provided if filter_func is not None.
    lower_cutoff : float, optional
        Lower cutoff wavenumber for the filter function. Must be provided if filter_func is not None.
    kwargs : dict, optional
        Additional keyword arguments.
        max_iteration : int, optional
            Maximum degree for the Taylor expansion. Default is 10.
        rmse_tolerance : float, optional
            Tolerance for the L2 norm. Default is 20.
        delta_emse_tolerance : float, optional
            Tolerance for the rmse ratio. Default is 1.008.
    
    ------------
    Returns:
    xarray.Dataset
        The result dataset will contain the following variables:
            ...
            - 'gravity': gravity data (xarray.DataArray)
            - 'inverted depth': inverted depth of the causative interface (xarray.DataArray)
               Each inverted depth atributed with its rmse.
    
    """
    
    density_contrast, reference_depth = hyperparam
    if reference_depth > 0:
        reference_depth *= -1
    hyperparam = (density_contrast, reference_depth)
    max_iteration = int(kwargs.get('max_iteration', 20))
    rmse_tolerance = kwargs.get('rmse_tolerance', 10.0)
    delta_rmse_tolerance = kwargs.get('delta_rmse_tolerance', 1.008)
    increase_limit = kwargs.get('increase_limit', 0.10)  # 10% increase limit

    util.grid_sanity_check(data)

    copied = copy.deepcopy(data)

    if isinstance(data, xarray.DataArray):
        observed_gravity = copied
        dataset = xarray.Dataset(
            {
                'gravity': copied,
            }
        )
    elif isinstance(data, xarray.Dataset):
        data_vars = [i for i in copied.data_vars.keys()] 
        if 'gravity' not in data_vars:
            msg1 = (
                "'gravity' variable was not found in the dataset. "
                "Is it available in a different name? [Y/N]"
            )
            ans1 = input(msg1)
            if ans1.lower() == 'y':
                msg2 = (
                    "Please provide the gravity grid name: "
                )
                ans2 = input(msg2)
                observed_gravity = copied[ans2]
                dataset = copied.rename_vars(
                    {
                        ans2: 'gravity',
                    }
                )
            else:
                raise ValueError("Gravity variable did not exist.")
        else:
            observed_gravity = copied['gravity']
            dataset = copied
    
    # observed_gravity = dataset['gravity']
    # L2_norms = []
    scores = []
    delta_scores = []

    for n in range(1, max_iteration + 1):

        backup = copy.deepcopy(dataset)
        grid = dataset['gravity']
        if (filter_func is not None) and (upper_cutoff is None or lower_cutoff is None):
            raise ValueError("Both upper_cutoff and lower_cutoff must be provided when using a filter function.")

        inverted_elev = inverse_func(
            grid,
            hyperparam,
            filter_func, # type: ignore
            upper_cutoff, # type: ignore
            lower_cutoff, # type: ignore
        )

        # Ensure observed_gravity is a DataArray, not a Dataset
        if isinstance(observed_gravity, xarray.Dataset):
            observed_gravity_da = observed_gravity['gravity']
        else:
            observed_gravity_da = observed_gravity

        dataset, score = update_misfit_and_gravity(
            observed_gravity_da,
            inverted_elev,
            hyperparam,
            n,
            forward_func,
        )

        if n == 1:
            scores.append(score)
            delta_score = numpy.inf
        else:
            if score < scores[-1]:
                scores.append(score)
                delta_score = scores[-2] / scores[-1]
                delta_scores.append(delta_score)
            else:
                dataset = backup
                break

        #Temporary breakpoint decision
        # if (rmse < rmse_tolerance) or (delta_rmse < delta_rmse_tolerance):
        #     break

        end, termination_reason = terminate(
            iteration_number = n,
            max_iterations = max_iteration,
            rmses = scores,
            rmse_tolerance = rmse_tolerance,
            delta_rmse = delta_score,
            previous_delta_rmse = delta_scores[-1] if delta_scores else numpy.inf,
            delta_rmse_tolerance = delta_rmse_tolerance,
            increase_limit = increase_limit,
        )

        if end:
            logger.info(
                f"Inversion terminated after {n} iterations due to {termination_reason}.\n"
                f"Final RMSE: {score:.4f} mGal."
            )
            break

    # Ensure the first return value is always a Dataset
    if isinstance(dataset, xarray.DataArray):
        dataset = xarray.Dataset({'result': dataset})
    return (dataset, scores, delta_scores)



        

        
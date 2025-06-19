"""
This is library for simplification of type hinting in pynatple.
"""

import xarray
from numpy.typing import NDArray
from typing import List, Optional, Callable, Tuple


# THE TAXONOMY
# The genome, It has two type:
BitString = str
Hyperparameter = Tuple[int | float, int | float]

# The Individual
Individual = BitString | Hyperparameter

# The Population
Population = List[Individual]

# And for shorter writing:
DepthData = xarray.DataArray | xarray.Dataset | List[xarray.DataArray]

# THE EVOLUTIONARY OPERATOR
PopulateFunc = Callable[[int, Hyperparameter, Hyperparameter, bool], Population]
FitnessFunc = Callable[[Individual, xarray.DataArray, Optional[DepthData]], float]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Individual, Individual]]
CrossoverFunc = Callable[[Individual, Individual, float, float], Tuple[Individual, Individual]]
MutationFunc = Callable[[Individual, Tuple[float, float], int, int, float], Individual]
PrinterFunc = Callable[[Population, int, List[float]], None]

# BASIC INVERSION OPERATOR
ForwardFunc = Callable[[xarray.DataArray, Hyperparameter], xarray.DataArray]
FilterFunc = Callable[[xarray.DataArray | NDArray, float, float], NDArray]
InverseFunc = Callable[[xarray.DataArray, Hyperparameter, FilterFunc, float, float], xarray.DataArray]
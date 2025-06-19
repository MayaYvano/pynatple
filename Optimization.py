import numpy
import xarray
import time
import copy
import logging

from pynatple import util, Inversion # type: ignore[self-owned module]
from random import choices, randint, randrange, uniform
from typing import List, Optional, Tuple, Any

from pynatple.pronounce import ( # type: ignore
    BitString,
    Hyperparameter,
    Individual,
    Population,
    DepthData,
    PopulateFunc,
    FitnessFunc,
    CrossoverFunc,
    MutationFunc,
    PrinterFunc,
)

# LOGGING STUFF
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.basicConfig(format = '%(name)s - %(levelname)s -> %(message)s')

# EVOLUTION FUNCTION
def generate_individual(
    lower_bound: Hyperparameter,
    upper_bound: Hyperparameter,
    binary:bool,
) -> Individual:
    
    random_density_contrast = randint(int(lower_bound[0]), int(upper_bound[0]))
    random_reference_depth = randint(int(upper_bound[1]), int(lower_bound[1]))
    random_hyperparameter = (random_density_contrast, random_reference_depth)

    if binary:
        return util.binary(random_hyperparameter)
    else:
        return random_hyperparameter


def generate_population(
    size: int,
    lower_bound: Hyperparameter = (0, 0),
    upper_bound: Hyperparameter = (1e3, -1e5),
    binary:bool = True,
) -> Population:
    
    if lower_bound == (0, 0) or upper_bound == (1e3, -1e5):
        logger.warning(
            'Initial estimation range not set. Be careful with the optimization final result.'
        )

    return [generate_individual(lower_bound, upper_bound, binary) for _ in range(size)]


def fitness(
    individual: Individual,
    data: xarray.DataArray,
    depth_control: DepthData,
    **kwargs:Any,
) -> float:
    """
    for depth control, if it is a scatter or regular plotted data, please assign it as a DataArray,
    if it is a seismic-2D-like data, assign it as a DataArray list or Dataset. And please make sure
    the coordinates is in easting and northing designation format.
    """
    
    forward_func = kwargs.get('forward_func', Inversion.parker_forward)
    inverse_func = kwargs.get('inverse_func', Inversion.parker_oldenburg_inversion)
    filter_func = kwargs.get('filter_func', None)
    upper_cutoff = kwargs.get('filter_upper_cutoff', None)
    lower_cutoff = kwargs.get('filter_lower_cutoff', None)

    if isinstance(individual, BitString):
        hyperparameter = util.unbinary(individual)
    else:
        hyperparameter = individual

    result, _, _ = Inversion.run_inversion(
        data = data,
        hyperparam = hyperparameter,
        forward_func = forward_func,
        inverse_func = inverse_func,
        filter_func = filter_func,
        upper_cutoff = upper_cutoff,
        lower_cutoff = lower_cutoff,
    )

    if depth_control is not None:
            if isinstance(depth_control, xarray.DataArray):
                eval_points = result.inverted_depth.sel(easting = depth_control.easting, 
                                                        northing = depth_control.northing, 
                                                        method = 'nearest')
                val = depth_control - eval_points.values

                return float(util.eval(val, metric = 'rmse'))
            
            elif isinstance(depth_control, xarray.Dataset):
                data_vars = [vars for vars in depth_control.data_vars.keys()]
                misfit = []
                for var in data_vars:
                    eval_points = util.extract_data(result.inverted_depth, depth_control[var])
                    val = depth_control[var] - eval_points.values
                    misfit.append(util.eval(val, metric = 'rmse'))

                return float(numpy.nanmean(misfit))
            
            elif isinstance(depth_control, list):
                misfit = []
                for da in depth_control:
                    eval_points = util.extract_data(result.inverted_depth, da)
                    val = da - eval_points.values
                    misfit.append(util.eval(val, metric = 'rmse'))

                return float(numpy.nanmean(misfit))
            
            else:
                raise TypeError("depth_control must be a Dataset or a list of DataArrays.")

    return result.attrs['evaluation_score']


def single_point_crossover(
    a: BitString,
    b: BitString,
    crossover_rate: float = 0.5,
    crossover_proportion: Optional[float] = 0.5,
) -> Tuple[Individual, Individual]:
    if len(a) != len(b):
        raise ValueError("Both individuals must have a same length")

    length = len(a)

    if length < 2:
        return a, b

    p = randint(1, length - 1)
    return a[0:p] + b[p:], b[0:p] + a[p:]


def crossover_from_paper(
    a: Individual,
    b: Individual,
    crossover_rate: float = 0.5,
    crossover_proportion: Optional[float] = 0.5,
) -> Tuple[Individual, Individual]:
    """
    This based on what shown in Yu et al. [2025].
    """

    if len(a) != len(b):
        raise ValueError("Both individuals must have a same length")

    # Ensure crossover_proportion is a float and not None
    if crossover_proportion is None:
        crossover_proportion = 0.5

    if crossover_rate < uniform(0, 1):
        return a, b
    else:
        # Ensure a and b are tuples of numbers, not bitstrings
        if isinstance(a, str):
            a = util.unbinary(a)
        if isinstance(b, str):
            b = util.unbinary(b)

        # X-over for density contrast
        c1 = float(a[0]) * crossover_proportion + float(b[0]) * (1 - crossover_proportion)
        c2 = float(b[0]) * crossover_proportion + float(a[0]) * (1 - crossover_proportion)
        # X-over for reference depth
        d1 = float(a[1]) * crossover_proportion + float(b[1]) * (1 - crossover_proportion)
        d2 = float(b[1]) * crossover_proportion + float(a[1]) * (1 - crossover_proportion)

        mode = choices(['dens', 'dept', 'both'])[0]
        if mode == 'dens':
            c = (int(c1), int(a[1]))
            d = (int(c2), int(b[1]))
        
        elif mode == 'dept':
            c = (int(a[0]), int(d1))
            d = (int(b[0]), int(d2))

        else:
            c = (int(c1), int(d1))
            d = (int(c2), int(d2))

    # If original inputs were bitstrings, return as bitstrings
    if isinstance(a, tuple) and isinstance(b, tuple):
        return c, d
    else:
        return util.binary(c), util.binary(d)


def mutation(
    individual: BitString,
    hyperparameter_range:Optional[Tuple[float, float]] = None,
    generation:Optional[int] = None,
    generation_limit:Optional[int] = None, 
    mutation_rate: float = 0.5,
    num: int = 1, 
) -> Individual:
    individual_ = list(map(int, individual))
    for _ in range(num):
        index = randrange(len(individual_))
        if uniform(0, 1) > mutation_rate:
            individual_[index] = individual_[index] 
        else: 
            individual_[index] = abs(individual_[index] - 1)
    return ''.join(map(str, individual_))


def mutation_from_paper(
    individual: Individual,
    hyperparameter_range:Tuple[float, float],
    generation:int,
    generation_limit:int,
    mutation_rate: float = 0.5,
) -> Individual:
    """
    This based on what shown in Yu et al. [2025].
    """
    
    if mutation_rate < uniform(0, 1):
        return individual
    else:
        if isinstance(individual, BitString):
            individual = util.unbinary(individual)

        r = uniform(-1, 1)
        p = uniform(0, 1)

        # Mutation over density contrast
        c = int(individual[0] + hyperparameter_range[0] * r * p**(generation / generation_limit))
        # Mutation over reference depth
        d = int(individual[1] + hyperparameter_range[1] * r * p**(generation / generation_limit))

        mode = choices(['dens', 'dept', 'both'])[0]
        if mode == 'dens':
            mutated = (int(c), int(individual[1]))
        
        elif mode == 'dept':
            mutated = (int(individual[0]), int(d))

        else:
            mutated = (int(c), int(d))

    if isinstance(individual, BitString):
        return util.binary(mutated)
    else:
        return mutated


def population_fitness(
    population: Population, 
    data: xarray.DataArray,
    fitness_func: FitnessFunc,
    depth_control: DepthData,
    **kwargs,
) -> List[float]:
    return [fitness_func(individual, data, depth_control, **kwargs) for individual in population]


def selection_pair(
    population: Population,
    method: str, 
    weights:List[float],
) -> Population:
    match method:
        case 'roulette_wheel':
            return choices(
                population = population,
                weights = [1 / weight for weight in weights],
                k = 2,
            )
        
        case _:
            msg = 'Meh, dont know that method.'
            raise TypeError(msg)


def sort_population(
    population: Population,
    data: xarray.DataArray,
    depth_control: DepthData,
    **kwargs,
) -> Population:
    return sorted(
        population,
        key = lambda individual: fitness(
            individual,
            data,
            depth_control,
            **kwargs,
        )
    )


def statsprint(
    sorted_population: Population, 
    generation_id: int,
    population_scores:List[float],
) -> None:
    
    print(f'GENERATION {generation_id}')
    print('=============================')

    avg_fit = sum(population_scores) / len(sorted_population)
    
    print(f'Avg. Fitness: {avg_fit:.2f}')

    most_valuable = sorted_population[0]
    less_valuable = sorted_population[-1]

    if isinstance(most_valuable and less_valuable, BitString):
        most_valuable = util.unbinary(most_valuable)
        less_valuable = util.unbinary(less_valuable)

    print(
        f'Best hyperparameter: {most_valuable[0]} kg/m^3 & {most_valuable[1]} m'
        f' --> RMSE: {population_scores[0]:.2f}'
    )

    print(
        f'Worst hyperparameter: {less_valuable[0]} kg/m^3 & {less_valuable[1]} m'
        f' --> RMSE: {population_scores[-1]:.2f}'
    )

    print('')


def run_evolution(
    data: xarray.DataArray,
    depth_control: xarray.Dataset | xarray.DataArray | List,
    population_size: int,
    generation_limit: int,
    rmse_criteria: float,
    crossover_rate: float = 0.5,
    crossover_proportion:float = 0.99,
    mutation_rate: float = 0.5,
    selection_method:str = 'roulette_wheel',
    binary:bool = False,
    populational_evaluation: bool = False,
    lower_bound: Hyperparameter = (300, -20000),
    upper_bound: Hyperparameter = (600, -40000),
    populate_func: PopulateFunc = generate_population,
    fitness_func: FitnessFunc = fitness, #type: ignore
    crossover_func: CrossoverFunc = crossover_from_paper,
    mutation_func: MutationFunc = mutation_from_paper,
    printer: Optional[PrinterFunc] = None,
    # If you want to get all generations instead, turn this to True:
    get_all_result: bool = False,
    **kwargs,
) -> Tuple[Population, int, Population]:
    
    # NO-DEPTH CONTROL WARNING
    if depth_control is None:
        logger.warning(
            'No depth control provided. The optimization will be based on gravity misfit.'
        )
    
    # INITIATION
    population = populate_func(population_size, lower_bound, upper_bound, binary)
    hyperparameter_range = (
        numpy.abs(upper_bound[0] - lower_bound[0]),
        numpy.abs(upper_bound[1] - upper_bound[1]),
    )

    generational_list = []
    hall_of_fame = []
    
    start = time.perf_counter()
    i = 0  # Ensure i is always defined
    for i in range(1, generation_limit + 1):
        population = sort_population(population, data, depth_control)
        generational_list.append(population)
        hall_of_fame.append(population[0])

        # ORIGINALLY BEING A BACKUP BUT NOW IT SERVE NO PURPOSE.
        # BUT I STILL HESITATE TO ERASE IT. IDK.
        clone = copy.deepcopy(population)

        scores = population_fitness(population, data, fitness_func, depth_control, **kwargs)

        if printer is not None:
            printer(population, i, scores)

        # BEST SCORE AS CRITERIA'S CHALLENGER!
        if populational_evaluation:
            challenger = sum(scores) / len(scores)
        else: # individual_evaluation
            challenger = scores[0]
        # GEN.-1 NOT ALLOWED TO PARTICIPATE BECAUSE IT IS A RANDOM GENERATION,
        # NOT A 'CYBER-DARWINIAN' PRODUCT.
        if i == 1:
            pass
        else:
            if challenger < rmse_criteria:
                break
            else:
                pass

        # EVOLUTION TIME! 
        next_generation = population[0:2]

        for _ in range(int(len(population) / 2) - 1):

            # Parental selection
            parents = selection_pair(
                population = population,
                method = selection_method,
                weights = scores,
            )

            # Crossover
            offspring_a, offspring_b = crossover_func(
                parents[0], 
                parents[1],
                crossover_rate,
                crossover_proportion,
            )

            # Mutation
            offspring_a = mutation_func(
                offspring_a,
                hyperparameter_range,
                i,
                generation_limit,
                mutation_rate,
            )
            offspring_b = mutation_func(
                offspring_b,
                hyperparameter_range,
                i,
                generation_limit,
                mutation_rate,
            )

            # Get the new generation
            next_generation.extend([offspring_a, offspring_b])

        population = next_generation

    end = time.perf_counter()

    # If the loop never ran, set i to 0
    if i == 0:
        i = 0

    # SORT THE HALL OF FAME
    hall_of_fame = sort_population(hall_of_fame, data, depth_control)
    
    # Convert it into something we can read
    if binary:
        population = util.unbinary(population)

    logger.info(
        f'Evolution time: {end - start:.2f}s with {i} generation(s)'
    )

    if get_all_result:
        return generational_list, i, hall_of_fame
    else:
        return population, i, hall_of_fame
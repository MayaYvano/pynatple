import xarray
import time
import copy
import logging

from pynatple import util, Inversion
from random import choices, randint, randrange, random, uniform
from typing import List, Optional, Callable, Tuple, Any

# THE TAXONOMY
# The genome, It has two type:
BitString = str
Hyperparameter = Tuple[int | float, int | float]

# The Individual
Individual = BitString | Hyperparameter

# The Population
Population = List[Individual]

# THE EVOLUTIONARY OPERATOR
PopulateFunc = Callable[[], Population]
FitnessFunc = Callable[[Individual, xarray.DataArray, Optional[xarray.DataArray]], float]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Individual, Individual]]
CrossoverFunc = Callable[[Individual, Individual], Tuple[Individual, Individual]]
MutationFunc = Callable[[Individual], Individual]
PrinterFunc = Callable[[Population, int, FitnessFunc], None]

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
    
    random_density_contrast = randint(lower_bound[0], upper_bound[0])
    random_reference_depth = randint(upper_bound[1], lower_bound[1])
    random_hyperparameter = (random_density_contrast, random_reference_depth)

    if binary:
        return util.binary(random_hyperparameter)
    else:
        return random_hyperparameter


def generate_population(
    size: int,
    lower_bound: Optional[Hyperparameter] = (0, 0),
    upper_bound: Optional[Hyperparameter] = (1e3, -1e5),
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
    depth_control: Optional[xarray.DataArray] = None,
    **kwargs:Any,
) -> float:
    
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
        misfit = depth_control - result.inverted_depth.values
        return util.eval(misfit, metric = 'rmse')

    return result.attrs['evaluation_score']


def depth_fitness(
    individual: Individual,
    data: xarray.DataArray,
    control: xarray.DataArray,
    **kwargs:Any,
) -> float:
    
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

    misfit = control - result.inverted_depth.values

    return util.eval(misfit, metric = 'rmse')


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
    
    if isinstance(a and b, BitString):
        a, b = util.unbinary(a), util.unbinary(b)

    if crossover_rate < uniform(0, 1):
        return a, b
    else:
        # X-over for density contrast
        c1 = a[0] * crossover_proportion + b[0] * (1 - crossover_proportion)
        c2 = b[0] * crossover_proportion + a[0] * (1 - crossover_proportion)
        # X-over for reference depth
        d1 = a[1] * crossover_proportion + b[1] * (1 - crossover_proportion)
        d2 = b[1] * crossover_proportion + a[1] * (1 - crossover_proportion)

        mode = choices(['dens', 'dept', 'both'])
        if mode == 'dens':
            c, d = (c1, a[1]), (c2, b[1])
        
        elif mode == 'dept':
            c, d = (a[0], d1), (b[0], d2)

        else:
            c, d = (c1, d1), (c2, d2)

    if isinstance(a and b, BitString):
        return util.binary(c), util.binary(d)
    else:
        return c, d         


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

    if isinstance(individual, BitString):
        individual = util.unbinary(individual)
    
    mode = choices(['dens', 'dept', 'both'])
    if mutation_rate < uniform(0, 1):
        return individual
    else:
        r = uniform(-1, 1)
        p = uniform(0, 1)
        # Mutation over density contrast
        c = int(individual[0] + hyperparameter_range[0] * r * p**(generation / generation_limit))
        # Mutation over reference depth
        d = int(individual[1] + hyperparameter_range[1] * r * p**(generation / generation_limit))

        if mode == 'dens':
            mutated = (c, individual[1])
        
        elif mode == 'dept':
            mutated = (individual[0], c)

        else:
            mutated = (c, d)

    if isinstance(individual, BitString):
        return util.binary(mutated)
    else:
        return mutated


def population_fitness(
    population: Population, 
    data:xarray.DataArray,
    depth_control:xarray.DataArray,
    fitness_func:FitnessFunc,
) -> List[float]:
    return [fitness_func(individual, data, depth_control) for individual in population]


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
    data:xarray.DataArray,
    depth_control:xarray.DataArray,
) -> Population:
    return sorted(
        population,
        key = lambda individual: fitness(
            individual,
            data,
            depth_control,
        ),
    )


# BROKEN, I GUESS.
def statslog(
    data: xarray.DataArray,
    population: Population, 
    population_fitnesses: float,
    generation_id: int,
    fitness_func: FitnessFunc,
    inversion_func,
) -> Individual:
    logger.info(
        f'GENERATION {generation_id}\n'
        '============================='
    )

    # print("=============")
    # print("Population: [%s]" % ", ".join([self.genome_to_string(individual) 
    #                                       for individual 
    #                                       in population]))
    
    avg_fit = population_fitnesses / len(population)
    
    logger.info(f'Avg. Fitness: {avg_fit:.2f}')

    sorted_population = sort_population(population, fitness_func)

    logger.info(
        f'Best hyperparameter: {util.binary_to_float(sorted_population[0])}'
        f' with RMSE: {fitness_func(
            sorted_population[0],
            data = data,
            forward_func = forward_func,
            inverse_func = inverse_func,
        ):.2f}'
    )

    logger.info(
        f'Worst hyperparameter: {util.binary_to_float(sorted_population[-1])}'
        f' with RMSE: {fitness_func(
            sorted_population[-1],
            data = data,
            forward_func = forward_func,
            inverse_func = inverse_func,
        ):.2f}\n'
        ' '
    )
    
    # print(
    #     "Best: %s (%f)" % (util.binary_to_float(sorted_population[0]), 
    #                         fitness_func(sorted_population[0])))
    # print("Worst: %s (%f)" % (util.binary_to_float(sorted_population[-1]),
    #                         fitness_func(sorted_population[-1])))
    # print("")

    return sorted_population[0]


def statsprint(
    sorted_population: Population, 
    population_scores:List[float],
    generation_id: int,
) -> Individual:
    
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

    return sorted_population[0]


def run_evolution(
    data:xarray.DataArray,
    depth_control,
    population_size:int,
    generation_limit: int,
    rmse_criteria: float,
    binary:bool = False,
    crossover_rate: float = 0.5,
    crossover_proportion:float = 0.5,
    mutation_rate: float = 0.5,
    selection_method:str = 'roulette_wheel',
    lower_bound: Optional[Hyperparameter] = (0.0, 0.0),
    upper_bound: Optional[Hyperparameter] = (1.0e3, -1.0e5),
    populate_func: PopulateFunc = generate_population,
    fitness_func: FitnessFunc = fitness,
    crossover_func: CrossoverFunc = crossover_from_paper,
    mutation_func: MutationFunc = mutation_from_paper,
    printer: Optional[PrinterFunc] = None
) -> Tuple[Population, int, List[float]]:
    
    population = populate_func(population_size, lower_bound, upper_bound, binary)
    hyperparameter_range = (
        upper_bound[0] - lower_bound[0],
        upper_bound[1] - upper_bound[1],
    )
    population_scores = []
    # hall_of_fame = []
    
    start = time.perf_counter()
    for i in range(1, generation_limit + 1):
        population = sort_population(population, data, depth_control)

        clone = copy.deepcopy(population)

        scores = population_fitness(population, data, depth_control, fitness_func)

        if printer is not None:
            printer(population, scores, i)

        if (sum(scores) / len(population)) < rmse_criteria:
            break
        
        if i == 1:
            population_scores.append(sum(scores) / len(population))
        #     hall_of_fame.extend(population)
        # else:
        #     for individual in population:
        #         for goat in hall_of_fame:
        #             if individual < goat:
        #                 hall_of_fame.append(individual)
            
        # if len(hall_of_fame) > 10:
        #     hall_of_fame = sort_population(hall_of_fame)[:10]


        # Initiate the next generation with this sequence
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

        # scores_ = sum(population_fitness(next_generation, data, depth_control, fitness_func)) / len(next_generation)

        # population_scores.append(scores_)

        # # Termination decision
        # if scores_ > population_scores[-1]:
        #     population = clone
        #     logger.warning(
        #         f'Optimization terminated after {i} generation(s) due to rising populational RMSE.'
        #     )
        #     break
        # else:
        #     population = next_generation
        population = next_generation

    end = time.perf_counter()

    # Convert it into something we can read
    if binary:
        population = util.unbinary(population)
        # hall_of_fame = util.unbinary(hall_of_fame)

    logger.info(
        f'Evolution time: {end - start:.2f}s with {i} generation(s)'
    )
    # logger.info(
    #     f'G.O.A.T -> {hall_of_fame[0][0]} kg/m^3 & {hall_of_fame[0][1]} m '
    #     f'[RMSE: fitness(hall_of_fame[0])]'
    # )

    return population, i, population_scores
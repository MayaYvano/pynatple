import numpy
import xarray
import pandas
import time
import copy
import logging
import concurrent.futures

from pynatple import util, Inversion # type: ignore[self-owned module]
from random import choices, randint, randrange, uniform
from typing import List, Optional, Tuple, Any

from pynatple.pronounce import ( # type: ignore
    BitString,
    Hyperparameter,
    Individual,
    Population,
    InverseFunc,
    PrinterFunc,
)

# LOGGING STUFF
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.basicConfig(format = '%(name)s - %(levelname)s -> %(message)s')

"""
Restructuriztion of Optimization module, packed as a class to acomodate another ML algorithm in the future
if by any chance it is needed. Corrently worked for a binary mode, and work perfectly but not benign. This
will be the standard module for optimization work after it completed.
"""

# PRINTER FUNCTION
# This function is used to print the statistics of the population at each generation.
# Not mandatory though.
def printer(
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


# OPTIMIZATION OPTION ONE: EVOLUTIONARY WAY
class Evolution:

    def __init__(
        self,
        data: xarray.DataArray,
        control: xarray.DataArray,
        lower_bound: Tuple[int, int],
        upper_bound: Tuple[int, int],
        population_size: int,
        max_generation: int,
        crossover_rate: float,
        crossover_proportion: float,
        mutation_rate: float,
        modeling_func: InverseFunc,
    ):
        self._data = data
        self._control = control
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._population_size = population_size
        self._max_generation = max_generation
        self._crossover_rate = crossover_rate
        self._crossover_proportion = crossover_proportion
        self._mutation_rate = mutation_rate
        self._modeling_func = modeling_func
        self.range = (
            numpy.abs(self._upper_bound[0] - self._lower_bound[0]),
            numpy.abs(self._upper_bound[1] - self._upper_bound[1]),
        )
    

    def birth(
        self,
    ) -> Individual:
        random_density_contrast = randint(int(self._lower_bound[0]), int(self._upper_bound[0]))
        random_reference_depth = randint(int(self._upper_bound[1]), int(self._lower_bound[1]))
        return (random_density_contrast, random_reference_depth)
    

    def initialize_population(
        self,
    ) -> Population:
        return [self.birth() for _ in range(self._population_size)]
    

    def fitness(
        self,
        a: Individual,
        **kwargs,
    ) -> float:  

        pre = self._modeling_func(self._data, a, **kwargs)
        eval_points = util.get_eval_points(pre, self._control)
        miss = self._control.values - (a[1] - eval_points).values
        return util.eval(miss, metric = 'rmse')
    

    # def single_point_crossover(
    #     self,
    #     a,
    #     b,
    # ):
    #     probability = self.mutation_rate
    #     if len(a) != len(b):
    #         raise ValueError("Both individuals must have a same length")

    #     length = len(a)

    #     if length < 2:
    #         return a, b

    #     p = randint(1, length - 1)
    #     return a[0:p] + b[p:], b[0:p] + a[p:]

    def xover(
        self,
        a: Individual,
        b: Individual,
    ) -> Tuple[Individual, Individual]:
        """
        This based on what shown in Yu et al. [2025].
        """

        if len(a) != len(b):
            raise ValueError("Both individuals must have a same length")

        # Ensure crossover_proportion is a float and not None
        if self._crossover_proportion is None:
            self._crossover_proportion = 0.5

        if self._crossover_rate < uniform(0, 1):
            return a, b
        else:
            # Ensure a and b are tuples of numbers, not bitstrings
            if isinstance(a, str):
                a = util.unbinary(a)
            if isinstance(b, str):
                b = util.unbinary(b)

            # X-over for density contrast
            c1 = float(a[0]) * self._crossover_proportion + float(b[0]) * (1 - self._crossover_proportion)
            c2 = float(b[0]) * self._crossover_proportion + float(a[0]) * (1 - self._crossover_proportion)
            # X-over for reference depth
            d1 = float(a[1]) * self._crossover_proportion + float(b[1]) * (1 - self._crossover_proportion)
            d2 = float(b[1]) * self._crossover_proportion + float(a[1]) * (1 - self._crossover_proportion)

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

        return c, d
    

    # def mutation(
    #     self,
    #     a,
    #     num: int = 1, 
    # ):
    #     a_ = list(map(int, a))
    #     for _ in range(num):
    #         index = randrange(len(a_))
    #         if uniform(0, 1) > self.mutation_rate:
    #             a_[index] = a_[index] 
    #         else: 
    #             a_[index] = abs(a_[index] - 1)
    #     return ''.join(map(str, a_))
    

    def mutate(
        self,
        a: Individual,
        generation: int,
    ) -> Individual:
        """
        This based on what shown in Yu et al. [2025].
        """
        
        if self._mutation_rate < uniform(0, 1):
            return a
        else:

            r = uniform(-1, 1)
            p = uniform(0, 1)

            # Mutation over density contrast
            c = int(a[0] + self.range[0] * r * p**(generation / self._max_generation))
            # Mutation over reference depth
            d = int(a[1] + self.range[1] * r * p**(generation / self._max_generation))

            mode = choices(['dens', 'dept', 'both'])[0]
            if mode == 'dens':
                mutated = (int(c), int(a[1]))
            
            elif mode == 'dept':
                mutated = (int(a[0]), int(d))

            else:
                mutated = (int(c), int(d))

        return mutated


    def selection_pair(
        self,
        population: Population,
        method: str, 
        weights: List[float],
        parent_num: int = 2,
    ) -> List[Individual]:
        match method:
            case 'roulette_wheel':
                return choices(
                    population = population,
                    weights = [1.0 / weight for weight in weights],
                    k = parent_num,
                )
            
            case _:
                msg = 'Meh, dont know that method.'
                raise TypeError(msg)


    def sort_population(
        self,
        population: Population,
        **kwargs,
    ) -> Population:
        return sorted(
            population,
            key = lambda a: self.fitness(a, **kwargs)
        )
    

    @property
    def crossover_rate(self) -> float:
        return self._crossover_rate

    @crossover_rate.setter
    def crossover_rate(self, value: float):
        if not (0 <= value <= 1):
            raise ValueError("Crossover rate must be between 0 and 1.")
        self._crossover_rate = value


    @property
    def crossover_proportion(self) -> float:
        return self._crossover_proportion    

    @crossover_proportion.setter
    def crossover_proportion(self, value: float):
        if not (0 <= value <= 1):
            raise ValueError("Crossover proportion must be between 0 and 1.")
        self._crossover_proportion = value


    @property
    def mutation_rate(self) -> float:
        return self._mutation_rate   

    @mutation_rate.setter
    def mutation_rate(self, value: float):
        if not (0 <= value <= 1):
            raise ValueError("Mutation rate must be between 0 and 1.")
        self._mutation_rate = value
    

    @property
    def max_generation(self) -> int:
        return self._max_generation
    
    @max_generation.setter
    def max_generation(self, value: int):
        if value <= 0:
            raise ValueError("Maximum generation must be a positive integer.")
        self._max_generation = value
    

    @property
    def population_size(self) -> int:
        return self._population_size
    
    @population_size.setter
    def population_size(self, value: int):
        if value <= 0:
            raise ValueError("Population size must be a positive integer.")
        self._population_size = value
    

    @property
    def lower_bound(self) -> Tuple[int, int]:
        return self._lower_bound
    
    @lower_bound.setter
    def lower_bound(self, value: Tuple[int, int]):
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError("Lower bound must be a tuple of two integers.")
        self._lower_bound = value
    
    @property
    def upper_bound(self) -> Tuple[int, int]:
        return self._upper_bound
    
    @upper_bound.setter
    def upper_bound(self, value: Tuple[int, int]):
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError("Upper bound must be a tuple of two integers.")
        self._upper_bound = value
    

    @property
    def modeling_func(self) -> InverseFunc:
        return self._modeling_func
    
    @modeling_func.setter
    def modeling_func(self, value: InverseFunc):
        if not callable(value):
            raise ValueError("Modeling function must be callable.")
        self._modeling_func = value
    

    @property
    def data(self) -> xarray.DataArray:
        return self._data
    
    @data.setter
    def data(self, value: xarray.DataArray):
        if not isinstance(value, xarray.DataArray):
            raise ValueError("Data must be an xarray DataArray.")
        self._data = value
    
    
    @property
    def control(self) -> xarray.DataArray:
        return self._control
    
    @control.setter
    def control(self, value: xarray.DataArray):
        if not isinstance(value, xarray.DataArray):
            raise ValueError("Control must be an xarray DataArray.")
        self._control = value

    
    def genetic_algorithm(
        self,
        convergence_criteria: float,
        selection_method: str = 'roulette_wheel',
        convergence_repetition: int = 3,
        convergence_tolerance: float = 0.01,
        populational_evaluation: bool = False,
        printer: Optional[PrinterFunc] = None,
        get_all_result: bool = False,
        parallel: bool = False,
    ):
        gen_list = []
        gen_fit = []
        hall_of_fame = {}

        population = self.initialize_population()
        start = time.perf_counter()

        i = 0
        for i in range(1, self._max_generation + 1):
            population = self.sort_population(population)
            gen_list.append(population)

            if parallel:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(self.fitness, individual) for individual in population]
                    scores = [future.result() for future in concurrent.futures.as_completed(futures)]
            else:
                scores = [self.fitness(individual) for individual in population]

            gen_fit.append(sum(scores) / len(scores))
            hall_of_fame.update({population[0]: scores[0]})

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
                if challenger < convergence_criteria:
                    break
                else:
                    pass
            
            # CHECK FOR CONVERGENCE
            if i > (convergence_repetition + 1):
                # Check if the last couple generations have converged
                last_gen_fits = list(hall_of_fame.values())[-(convergence_repetition + 1):]
                if all(
                    abs(val - challenger) < convergence_tolerance 
                    for val in last_gen_fits
                ):
                    print(f'Convergence reached at generation {i}')
                    break
                else:
                    pass
            
            # MAKE A BACKUP PLAN
            # UNUSED, BUT HESITATE TO REMOVE IT.
            clone = copy.deepcopy(population)

            # EVOLUTION TIME! 
            next_generation = population[0:2]

            for _ in range(int(len(population) / 2) - 1):

                if numpy.nan in scores or 0 in scores:
                    print("NaN found in scores, breaking the loop.")
                    pass

                # Parental selection
                parents = self.selection_pair(
                    population = population,
                    method = selection_method,
                    weights = scores,
                )

                # Crossover
                offspring_a, offspring_b = self.xover(
                    a = parents[0], 
                    b = parents[1],
                )

                # Mutation
                offspring_a = self.mutate(
                    offspring_a,
                    generation=i,
                )
                offspring_b = self.mutate(
                    offspring_b,
                    generation=i,
                )

                # Get the new generation
                next_generation.extend([offspring_a, offspring_b])

            population = next_generation

        end = time.perf_counter()

        # If the loop never ran, set i to 0
        if i == 0:
            i = 0

        # # SORT THE HALL OF FAME
        # hall_of_fame = self.sort_population(hall_of_fame)
        
        # Convert it into something we can read
        # population = [util.unbinary(individual) for individual in population]
        # hall_of_fame = [util.unbinary(goat) for goat in hall_of_fame]

        logger.info(
            f'Evolution time: {end - start:.2f}s with {i} generation(s)'
        )
        
        if get_all_result:
            return gen_list, gen_fit, hall_of_fame
        else:
            return population, gen_fit, hall_of_fame
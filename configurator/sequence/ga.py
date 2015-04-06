"""Configuration dialogs based on genetic algorithms.
"""

import logging

import numpy as np
from pyeasyga.pyeasyga import GeneticAlgorithm

from . import PermutationDialogBuilder, PermutationDialog


__all__ = ["GADialogBuilder"]


log = logging.getLogger(__name__)


class GADialogBuilder(PermutationDialogBuilder):
    """Build a configuration dialog using genetic algorithms.

    Arguments:
        total_episodes: Total number of simulated episodes.
        consistency: Type of consistency check used to filter the
            domain of the remaining questions during the simulation of
            the episodes. Possible values are: `'global'` and `'local'`.
            This argument is ignored for rule-based dialogs.
        eval_episodes: Number of episodes simulated for the evaluation
            of the fitness of a permutation sequence.
        initial_solution: Method used to generate each initial
            permutation sequence. Possible values are: `'random'`
            (start from a random permutation) and `'degree'` (use an
            initialization heuristic based on degree of the variables
            in the constraint network or the times it appears on the
            left-hand-side of a rule).
         population_size: Population size.
         mutation_prob: Mutation probability.
         tournament_size: Tournament selection size.

    See :class:`configurator.dialogs.DialogBuilder` for the remaining
    arguments.
    """

    def __init__(self, var_domains, sample=None, rules=None, constraints=None,
                 total_episodes=1000,
                 consistency="local",
                 eval_episodes=30,
                 initial_solution="random",
                 population_size=50,
                 mutation_prob=0.2,
                 tournament_size=2,
                 validate=False):
        super().__init__(var_domains, sample, rules, constraints,
                         total_episodes, consistency, eval_episodes,
                         initial_solution, validate)
        self._population_size = population_size
        self._mutation_prob = mutation_prob
        self._tournament_size = tournament_size

    def build_dialog(self):
        """Construct a configuration dialog.

        Returns:
            An instance of `configurator.dialogs.Dialog` subclass.
        """
        num_generations = (self._total_episodes //
                           self._eval_episodes //
                           self._population_size)
        ga = GeneticAlgorithm(None, population_size=self._population_size,
                              generations=num_generations,
                              crossover_probability=1.0,
                              mutation_probability=self._mutation_prob,
                              elitism=True, maximise_fitness=False)
        ga.create_individual = self._create_individual
        ga.crossover_function = self._crossover
        ga.mutate_function = self._mutate_var_perm
        ga.selection_function = ga.tournament_selection
        ga.tournament_size = self._tournament_size
        ga.fitness_function = self._fitness
        log.info("running the GA for %d generations", num_generations)
        ga.run()
        mean_questions, var_perm = ga.best_individual()
        dialog = PermutationDialog(self.var_domains, var_perm,
                                   self.rules, self.constraints,
                                   validate=self._validate)
        return dialog

    def _create_individual(self, data=None):
        if self._initial_solution == "random":
            return self._generate_random_var_perm()
        elif self._initial_solution == "degree":
            return self._generate_degree_var_perm()

    def _crossover(self, parent1, parent2):
        # Precedence Preservative Crossover (PPX).
        children = [[], []]
        for i in range(2):
            p1, p2 = parent1[:], parent2[:]
            for from_p1 in np.random.choice([True, False], len(p1)):
                elem = p1[0] if from_p1 else p2[0]
                children[i].append(elem)
                p1.remove(elem)
                p2.remove(elem)
        return children

    def _fitness(self, individual, data=None):
        return self._eval_var_perm(individual)

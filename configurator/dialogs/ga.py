#
#  Copyright 2015 Yasser Gonzalez Fernandez
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#

"""Configuration dialogs based on genetic algorithms.
"""

import logging
import random
import pprint
import collections
import zipfile

import dill
import numpy as np
from pyeasyga.pyeasyga import GeneticAlgorithm

from ..dialogs import DialogBuilder, Dialog


__all__ = ["GADialogBuilder"]


log = logging.getLogger(__name__)


class GADialogBuilder(DialogBuilder):
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
            in the constraint network or the times they appear on the
            left-hand-side of a rule).
        population_size: Population size.
        mutation_prob: Mutation probability.
        tournament_size: Tournament selection size.

    See :class:`configurator.dialogs.DialogBuilder` for the remaining
    arguments.
    """

    def __init__(self, var_domains, sample=None, rules=None, constraints=None,
                 total_episodes=30000,
                 consistency="local",
                 eval_episodes=30,
                 initial_solution="random",
                 population_size=20,
                 mutation_prob=0.2,
                 tournament_size=2,
                 validate=False):
        super().__init__(var_domains, sample, rules, constraints, validate)
        if consistency not in {"global", "local"}:
            raise ValueError("Invalid consistency value")
        if initial_solution not in {"random", "degree"}:
            raise ValueError("Invalid initial_solution value")
        self._total_episodes = total_episodes
        self._consistency = consistency
        self._eval_episodes = eval_episodes
        self._initial_solution = initial_solution
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
        ga.fitness_function = self._eval_var_perm
        log.info("running the GA for %d generations", num_generations)
        ga.run()
        mean_questions, var_perm = ga.best_individual()
        log.info("best average number of questions %g", mean_questions)
        dialog = GADialog(self.var_domains, var_perm,
                          self.rules, self.constraints,
                          validate=self._validate)
        return dialog

    def _create_individual(self, data=None):
        if self._initial_solution == "random":
            return self._generate_random_var_perm()
        elif self._initial_solution == "degree":
            return self._generate_degree_var_perm()

    def _generate_random_var_perm(self):
        var_perm = list(range(len(self.var_domains)))
        random.shuffle(var_perm)
        return var_perm

    def _generate_degree_var_perm(self):
        var2tier = collections.Counter()
        if self.rules:
            # Variables that appear the most in the LHS go first.
            reverse_tiers = True
            for rule in self.rules:
                for var_index in rule.lhs.keys():
                    var2tier[var_index] += 1
        else:
            # Variables that participate in many constraints go last.
            reverse_tiers = False
            for var_indices, constraint_fun in self.constraints:
                for var_index in var_indices:
                    var2tier[var_index] += 1
        tier2vars = collections.defaultdict(list)
        for var_index, tier in var2tier.items():
            tier2vars[tier].append(var_index)
        var_perm = []
        for var_tier in sorted(tier2vars.keys(), reverse=reverse_tiers):
            random.shuffle(tier2vars[var_tier])  # break ties randomly
            var_perm.extend(tier2vars[var_tier])
        return var_perm

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

    def _mutate_var_perm(self, var_perm):
        # Randomly swap two questions.
        i = random.randint(0, len(var_perm) - 1)
        j = random.randint(0, len(var_perm) - 1)
        var_perm[i], var_perm[j] = var_perm[j], var_perm[i]

    def _eval_var_perm(self, var_perm, data=None):
        log.debug("evaluating permutation:\n%s", pprint.pformat(var_perm))
        log.info("simulating %d episodes", self._eval_episodes)
        dialog = GADialog(self.var_domains, var_perm,
                          self.rules, self.constraints)
        num_questions = []
        for i in range(self._eval_episodes):
            result = self._simulate_dialog(dialog)
            if result is None:
                log.debug("finished an episode, reached an inconsistent state")
            else:
                log.debug("finished an episode, asked %d questions", result)
                num_questions.append(result)
        mean_questions = np.mean(num_questions)
        log.info("finished %d complete episodes, average number of questions %g",
                 len(num_questions), mean_questions)
        return mean_questions

    def _simulate_dialog(self, dialog):
        num_questions = 0
        dialog.reset()
        while dialog.is_consistent() and not dialog.is_complete():
            var_index = dialog.get_next_question()
            # Simulate the user response.
            var_values = dialog.get_possible_answers(var_index)
            probs = np.empty_like(var_values, dtype=np.float)
            for i, var_value in enumerate(var_values):
                response = {var_index: var_value}
                probs[i] = self._freq_table.cond_prob(response, dialog.config)
            bins = np.cumsum(probs / probs.sum())
            sampled_bin = np.random.random((1, ))
            var_value = var_values[int(np.digitize(sampled_bin, bins))]
            # Give the answer back to the dialog.
            dialog.set_answer(var_index, var_value, self._consistency)
            num_questions += 1
        return num_questions if dialog.is_consistent() else None


class GADialog(Dialog):
    """Configuration dialog based on permutation sequences.

    Arguments:
        var_perm: A list containing a permutation of the variables.

    See :class:`configurator.dialogs.Dialog` for information about the
    remaining arguments, attributes and methods.
    """

    def __init__(self, var_domains, var_perm,
                 rules=None, constraints=None, validate=False):
        self.var_perm = var_perm
        super().__init__(var_domains, rules, constraints, validate)

    def _validate(self):
        if set(self.var_perm) != set(range(len(self.var_domains))):
            raise ValueError("Invalid var_perm value")

    def reset(self):
        super().reset()
        self._curr_var_index = 0

    def get_next_question(self):
        if not self.is_complete():
            while self.var_perm[self._curr_var_index] in self.config:
                self._curr_var_index += 1
        next_question = self.var_perm[self._curr_var_index]
        return next_question

    def save(self, file_path):
        with zipfile.ZipFile(file_path, "w") as zip_file:
            zip_file.writestr("__class__", dill.dumps(self.__class__))
            zip_file.writestr("var_domains", dill.dumps(self.var_domains))
            zip_file.writestr("rules", dill.dumps(self.rules))
            zip_file.writestr("constraints", dill.dumps(self.constraints))
            zip_file.writestr("var_perm", dill.dumps(self.var_perm))

    @classmethod
    def load(cls, zip_file):
        var_domains = dill.loads(zip_file.read("var_domains"))
        rules = dill.loads(zip_file.read("rules"))
        constraints = dill.loads(zip_file.read("constraints"))
        var_perm = dill.loads(zip_file.read("var_perm"))
        return GADialog(var_domains, var_perm, rules, constraints)

"""Configuration dialogs based on optimization.
"""

import sys
import math
import random
import pprint
import logging
import collections

import numpy as np
from simanneal import Annealer

from .dialogs import DialogBuilder, PermutationDialog


__all__ = ["OptimDialogBuilder"]


log = logging.getLogger(__name__)


class OptimDialogBuilder(DialogBuilder):
    """Build a configuration dialog using optimization.
    """

    def __init__(self, var_domains, sample, rules=None, constraints=None,
                 consistency="local",
                 num_episodes=1000,
                 eval_batch=30,
                 validate=False):
        super().__init__(var_domains, sample, rules, constraints, validate)
        if consistency not in {"global", "local"}:
            raise ValueError("Invalid consistency value")
        self._consistency = consistency
        self._num_episodes = num_episodes
        self._eval_batch = eval_batch

    def build_dialog(self):
        """Construct a configuration dialog.

        Returns:
            An instance of `configurator.dialogs.Dialog` subclass.
        """
        annealer = DialogAnnealer(self.var_domains, self._freq_table,
                                  self.rules, self.constraints,
                                  self._consistency,
                                  self._num_episodes,
                                  self._eval_batch)
        var_perm, median_num_questions = annealer.anneal()
        dialog = PermutationDialog(self.var_domains, var_perm,
                                   self.rules, self.constraints,
                                   validate=self._validate)
        return dialog


class DialogAnnealer(Annealer):

    def __init__(self, var_domains, freq_table, rules, constraints,
                 consistency, num_episodes, eval_batch):
        self._var_domains = var_domains
        self._freq_table = freq_table
        self._rules = rules
        self._constraints = constraints
        self._consistency = consistency
        self._num_episodes = num_episodes
        self._eval_batch = eval_batch
        # Annealer class initialization:
        max_energy_diff = len(self._var_domains) - 1
        self.Tmax = - max_energy_diff / math.log(0.8)
        self.Tmin = (- max_energy_diff /
                     math.log(math.sqrt(sys.float_info.epsilon)))
        self.steps = self._num_episodes // self._eval_batch
        self.updates = 0
        self.copy_strategy = "slice"
        super().__init__(self.initial_state())

    def initial_state(self):
        if self._rules or self._constraints:
            var_degrees = collections.Counter()
            if self._rules:
                for rule in self._rules:
                    for var_index in rule.lhs.keys():
                        var_degrees[var_index] += 1
            else:
                for var_indices, constraint_fun in self._constraints:
                    for var_index in var_indices:
                        var_degrees[var_index] += 1
            var_degrees = zip(var_degrees.values(), var_degrees.keys())
            var_perm = list(k for v, k in sorted(var_degrees, reverse=True))
        else:
            var_perm = list(range(len(self._var_domains)))
        return var_perm

    def move(self):
        # Randomly swap two questions.
        i = random.randint(0, len(self.state) - 1)
        j = random.randint(0, len(self.state) - 1)
        self.state[i], self.state[j] = self.state[j], self.state[i]

    def energy(self):
        # Objective function.
        log.debug("evaluating permutation:\n%s", pprint.pformat(self.state))
        log.info("simulating %d episodes", self._eval_batch)
        dialog = PermutationDialog(self._var_domains, self.state,
                                   self._rules, self._constraints)
        num_questions = []
        for i in range(self._eval_batch):
            num_questions.extend(self._simulate_dialog(dialog))
        median_num_questions = np.median(num_questions)
        log.info("finished %d complete episodes, median number of questions %g",
                 len(num_questions), median_num_questions)
        return median_num_questions

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
        if dialog.is_consistent():
            log.debug("finished an episode, asked %d questions", num_questions)
            return [num_questions]
        else:
            log.debug("finished an episode, reached an inconsistent state")
            return []

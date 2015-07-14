#
#  Copyright 2015 Yasser Gonzalez Fernandez
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#

"""Configuration dialogs based on permutation sequences.
"""

import random
import logging
import pprint
import collections
import zipfile

import numpy as np
import dill

from ..dialogs import DialogBuilder, Dialog


__all__ = ["SADialogBuilder", "GADialogBuilder"]


log = logging.getLogger(__name__)


class SequenceDialogBuilder(DialogBuilder):
    """Configuration dialog builder based on permutation sequences.

    This is the base class of all the configuration dialog builders
    defined in :mod:`configurator.sequence`. It contains supporting
    methods shared by the different subclasses.
    """

    def __init__(self, var_domains, sample, rules, constraints,
                 total_episodes, consistency, eval_episodes,
                 initial_solution, validate):
        super().__init__(var_domains, sample, rules, constraints, validate)
        if consistency not in {"global", "local"}:
            raise ValueError("Invalid consistency value")
        if initial_solution not in {"random", "degree"}:
            raise ValueError("Invalid initial_solution value")
        self._total_episodes = total_episodes
        self._consistency = consistency
        self._eval_episodes = eval_episodes
        self._initial_solution = initial_solution

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

    def _mutate_var_perm(self, var_perm):
        # Randomly swap two questions.
        i = random.randint(0, len(var_perm) - 1)
        j = random.randint(0, len(var_perm) - 1)
        var_perm[i], var_perm[j] = var_perm[j], var_perm[i]

    def _eval_var_perm(self, var_perm):
        log.debug("evaluating permutation:\n%s", pprint.pformat(var_perm))
        log.info("simulating %d episodes", self._eval_episodes)
        dialog = SequenceDialog(self.var_domains, var_perm,
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


class SequenceDialog(Dialog):
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
        return SequenceDialog(var_domains, var_perm, rules, constraints)


# Keep this in the end, .sequence and .sequence.sa import each other.
from .ga import GADialogBuilder

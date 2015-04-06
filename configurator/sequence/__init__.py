"""Configuration dialogs based on permutation sequences.
"""

import random
import logging
import pprint
import collections

import numpy as np

from ..dialogs import DialogBuilder, Dialog


__all__ = ["SADialogBuilder"]


log = logging.getLogger(__name__)


class PermutationDialogBuilder(DialogBuilder):
    """Configuration dialog builder based on permutation sequences.

    This is the base class of all the configuration dialog builders
    defined in :mod:`configurator.sequence`. It contains supporting
    methods shared by the different subclasses.
    """

    def __init__(self, var_domains, sample, rules, constraints,
                 total_episodes, consistency, eval_episodes,
                 initialization, validate):
        super().__init__(var_domains, sample, rules, constraints, validate)
        if consistency not in {"global", "local"}:
            raise ValueError("Invalid consistency value")
        if initialization not in {"random", "degree"}:
            raise ValueError("Invalid initialization value")
        self._total_episodes = total_episodes
        self._consistency = consistency
        self._eval_episodes = eval_episodes
        self._initialization = initialization

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
        dialog = PermutationDialog(self.var_domains, var_perm,
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
        log.info("finished %d complete episodes, mean number of questions %g",
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


class PermutationDialog(Dialog):
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


# Keep this in the end. .sequence and .sequence.sa import each other.
from .sa import SADialogBuilder

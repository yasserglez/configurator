"""Configuration dialogs based on simulated annealing.
"""

import sys
import math
import logging

from simanneal import Annealer

from . import PermutationDialogBuilder, PermutationDialog


__all__ = ["SADialogBuilder"]


log = logging.getLogger(__name__)


class SADialogBuilder(PermutationDialogBuilder):
    """Build a configuration dialog using simulated annealing.

    Arguments:
        total_episodes: Total number of simulated episodes.
        consistency: Type of consistency check used to filter the
            domain of the remaining questions during the simulation of
            the episodes. Possible values are: `'global'` and `'local'`.
            This argument is ignored for rule-based dialogs.
        eval_episodes: Number of episodes simulated for the evaluation
            of the fitness of a permutation sequence.
        initialization: Method used to generate the initial
            permutation sequence. Possible values are: `'random'`
            (start from a random permutation) and `'degree'` (use an
            initialization heuristic based on degree of the variables
            in the constraint network or the times it appears on the
            left-hand-side of a rule).

    See :class:`configurator.dialogs.DialogBuilder` for the remaining
    arguments.
    """

    def __init__(self, var_domains, sample=None, rules=None, constraints=None,
                 total_episodes=1000,
                 consistency="local",
                 eval_episodes=30,
                 initialization="random",
                 validate=False):
        super().__init__(var_domains, sample, rules, constraints,
                         total_episodes, consistency, eval_episodes,
                         initialization, validate)

    def build_dialog(self):
        """Construct a configuration dialog.

        Returns:
            An instance of `configurator.dialogs.Dialog` subclass.
        """
        generate_var_perm = (self._generate_degree_var_perm
                             if self._initialization == "degree" else
                             self._generate_random_var_perm)
        annealer = DialogAnnealer(self.var_domains,
                                  generate_var_perm,
                                  self._mutate_var_perm,
                                  self._eval_var_perm,
                                  self._total_episodes // self._eval_episodes)
        var_perm, mean_questions = annealer.anneal()
        dialog = PermutationDialog(self.var_domains, var_perm,
                                   self.rules, self.constraints,
                                   validate=self._validate)
        return dialog


class DialogAnnealer(Annealer):

    def __init__(self, var_domains, generate_var_perm,
                 mutate_var_perm, eval_var_perm, steps):
        self._generate_var_perm = generate_var_perm
        self._mutate_var_perm = mutate_var_perm
        self._eval_var_perm = eval_var_perm
        # Annealer class initialization:
        super().__init__(self._generate_var_perm())
        self.steps = steps
        max_energy_diff = len(var_domains) - 1
        self.Tmax = - max_energy_diff / math.log(0.8)
        self.Tmin = (- max_energy_diff /
                     math.log(math.sqrt(sys.float_info.epsilon)))
        log.info("initial temp %g, final temp %g", self.Tmax, self.Tmin)
        self.copy_strategy = "slice"
        self.updates = 0

    def move(self):
        self._mutate_var_perm(self.state)

    def energy(self):
        return self._eval_var_perm(self.state)

import numpy as np
from numpy.testing import assert_array_equal, assert_raises

from ..dp import (DPDialogBuilder, MDP, EpisodicMDP,
                  PolicyIteration, ValueIteration)

from .common import BaseTest, load_email_client, load_grid_world


class TestDPDialogBuilder(BaseTest):

    def setup(self):
        super().setup()
        self._email_client = load_email_client()

    def _test_builder(self, algorithm, discard_states,
                      partial_assoc_rules, aggregate_terminals):
        builder = DPDialogBuilder(
            config_sample=self._email_client.config_sample,
            validate=True,
            assoc_rule_algorithm="apriori",
            assoc_rule_min_support=self._email_client.min_support,
            assoc_rule_min_confidence=self._email_client.min_confidence,
            dp_algorithm=algorithm,
            dp_discard_states=discard_states,
            dp_partial_assoc_rules=partial_assoc_rules,
            dp_aggregate_terminals=aggregate_terminals)
        dialog = builder.build_dialog()
        for var_index in self._email_client.questions:
            assert dialog.get_next_question() == var_index
            dialog.set_answer(var_index, self._email_client.config[var_index])
        assert dialog.is_complete()
        assert dialog.config == self._email_client.config

    def _test_builder_without_optim(self, algorithm):
        self._test_builder(algorithm, False, False, False)

    def _test_builder_with_optim(self, algorithm):
        self._test_builder(algorithm, True, True, True)

    def test_value_iteration_without_optim(self):
        self._test_builder_without_optim("value-iteration")

    def test_policy_iteration_without_optim(self):
        self._test_builder_without_optim("policy-iteration")

    def test_value_iteration_with_optim(self):
        self._test_builder_with_optim("value-iteration")

    def test_policy_iteration_with_optim(self):
        self._test_builder_with_optim("policy-iteration")


grid_world = load_grid_world()
S, A = grid_world.num_states, grid_world.num_actions
S0, Sn = grid_world.initial_state, grid_world.terminal_state
P = grid_world.transitions
R = grid_world.rewards
GAMMA = grid_world.discount_factor
POLICY = grid_world.policy
del grid_world


class TestMDP(BaseTest):

    mdp_class = MDP

    def setup(self):
        super().setup()

    def test_init_attributes(self):
        mdp = self.mdp_class(P, R, GAMMA)
        assert_array_equal(mdp.transitions, P)
        assert_array_equal(mdp.rewards, R)
        assert mdp.discount_factor == GAMMA

    def test_init_validation(self):
        invalid_P = np.random.rand(A, S, S)
        assert_raises(ValueError, self.mdp_class, invalid_P, R, GAMMA)
        invalid_R = np.zeros((A - 1, S, S))
        assert_raises(ValueError, self.mdp_class, P, invalid_R, GAMMA)
        invalid_gamma = 2
        assert_raises(ValueError, self.mdp_class, P, R, invalid_gamma)


class TestEpisodicMDP(TestMDP):

    mdp_class = EpisodicMDP

    def test_init_attributes(self):
        mdp = self.mdp_class(P, R, GAMMA, S0, Sn)
        assert mdp.initial_state == S0
        assert mdp.terminal_state == Sn

    def test_init_validation(self):
        super().test_init_validation()
        invalid_P = np.random.rand(A, S, S)
        assert_raises(ValueError, self.mdp_class,
                      invalid_P, R, GAMMA, S0, Sn)
        invalid_R = np.random.rand(A, S, S)
        assert_raises(ValueError, self.mdp_class,
                      P, invalid_R, GAMMA, S0, Sn)


class BaseTestMDPSolver(BaseTest):

    solver_class = None

    def setup(self):
        super().setup()
        self.solver = self.solver_class()

    def test_solve(self):
        mdp = EpisodicMDP(P, R, GAMMA)
        policy = self.solver.solve(mdp)
        assert all([policy[i] == POLICY[i] for i in range(len(POLICY))])


class TestPolicyIteration(BaseTestMDPSolver):

    solver_class = PolicyIteration


class TestValueIteration(BaseTestMDPSolver):

    solver_class = ValueIteration

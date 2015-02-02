import numpy as np
from numpy.testing import assert_array_equal, assert_raises

from configurator.dp import (DPDialogBuilder, MDP, EpisodicMDP,
                             PolicyIteration, ValueIteration)


class TestDPDialogBuilder(object):

    def _test_builder(self, algorithm, discard_states,
                      partial_assoc_rules, aggregate_terminals,
                      email_client):
        builder = DPDialogBuilder(
            config_sample=email_client.config_sample,
            validate=True,
            assoc_rule_min_support=email_client.min_support,
            assoc_rule_min_confidence=email_client.min_confidence,
            dp_algorithm=algorithm,
            dp_discard_states=discard_states,
            dp_partial_assoc_rules=partial_assoc_rules,
            dp_aggregate_terminals=aggregate_terminals)
        dialog = builder.build_dialog()
        for var_index in email_client.questions:
            assert dialog.get_next_question() == var_index
            dialog.set_answer(var_index, email_client.config[var_index])
        assert dialog.is_complete()
        assert dialog.config == email_client.config

    def _test_builder_without_optim(self, algorithm, email_client):
        self._test_builder(algorithm, False, False, False, email_client)

    def _test_builder_with_optim(self, algorithm, email_client):
        self._test_builder(algorithm, True, True, True, email_client)

    def test_value_iteration_without_optim(self, email_client):
        self._test_builder_without_optim("value-iteration", email_client)

    def test_policy_iteration_without_optim(self, email_client):
        self._test_builder_without_optim("policy-iteration", email_client)

    def test_value_iteration_with_optim(self, email_client):
        self._test_builder_with_optim("value-iteration", email_client)

    def test_policy_iteration_with_optim(self, email_client):
        self._test_builder_with_optim("policy-iteration", email_client)


class TestMDP(object):

    mdp_class = MDP

    def test_init_attributes(self, grid_world):
        P = grid_world.transitions
        R = grid_world.rewards
        gamma = grid_world.discount_factor
        mdp = self.mdp_class(P, R, gamma)
        assert_array_equal(mdp.transitions, P)
        assert_array_equal(mdp.rewards, R)
        assert mdp.discount_factor == gamma

    def test_init_validation(self, grid_world):
        P = grid_world.transitions
        R = grid_world.rewards
        gamma = grid_world.discount_factor
        S = grid_world.num_states
        A = grid_world.num_actions
        invalid_P = np.random.rand(A, S, S)
        assert_raises(ValueError, self.mdp_class, invalid_P, R, gamma)
        invalid_R = np.zeros((A - 1, S, S))
        assert_raises(ValueError, self.mdp_class, P, invalid_R, gamma)
        invalid_gamma = 2
        assert_raises(ValueError, self.mdp_class, P, R, invalid_gamma)


class TestEpisodicMDP(TestMDP):

    mdp_class = EpisodicMDP

    def test_init_attributes(self, grid_world):
        P = grid_world.transitions
        R = grid_world.rewards
        gamma = grid_world.discount_factor
        S0 = grid_world.initial_state
        Sn = grid_world.terminal_state
        mdp = self.mdp_class(P, R, gamma, S0, Sn)
        assert mdp.initial_state == S0
        assert mdp.terminal_state == Sn

    def test_init_validation(self, grid_world):
        super().test_init_validation(grid_world)
        P = grid_world.transitions
        R = grid_world.rewards
        gamma = grid_world.discount_factor
        S = grid_world.num_states
        S0 = grid_world.initial_state
        Sn = grid_world.terminal_state
        A = grid_world.num_actions
        invalid_P = np.random.rand(A, S, S)
        assert_raises(ValueError, self.mdp_class,
                      invalid_P, R, gamma, S0, Sn)
        invalid_R = np.random.rand(A, S, S)
        assert_raises(ValueError, self.mdp_class,
                      P, invalid_R, gamma, S0, Sn)


class BaseTestMDPSolver(object):

    solver_class = None

    def setup(self):
        self.solver = self.solver_class()

    def test_solve(self, grid_world):
        P = grid_world.transitions
        R = grid_world.rewards
        gamma = grid_world.discount_factor
        mdp = EpisodicMDP(P, R, gamma)
        policy = self.solver.solve(mdp)
        assert all([policy[i] == grid_world.policy[i]
                    for i in range(len(grid_world.policy))])


class TestPolicyIteration(BaseTestMDPSolver):

    solver_class = PolicyIteration


class TestValueIteration(BaseTestMDPSolver):

    solver_class = ValueIteration

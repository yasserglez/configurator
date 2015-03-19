import pytest

import numpy as np
from numpy.testing import assert_raises

from configurator.dp import DPDialogBuilder, EpisodicMDP


class TestDPDialogBuilder(object):

    def _test_builder(self, email_client, with_improvements):
        builder = DPDialogBuilder(email_client.var_domains,
                                  email_client.sample,
                                  rules=email_client.rules,
                                  discard_states=with_improvements,
                                  partial_rules=with_improvements,
                                  aggregate_terminals=with_improvements,
                                  validate=True)
        dialog = builder.build_dialog()
        for config, num_questions in email_client.scenarios:
            dialog.reset()
            for i in range(num_questions):
                var_index = dialog.get_next_question()
                dialog.set_answer(var_index, config[var_index])
            assert dialog.is_complete()
            assert dialog.config == config

    @pytest.mark.parametrize("with_improvements", (True, False),
                             ids=("with_improvements", "without_improvements"))
    def test_build_dialog(self, email_client, with_improvements):
        self._test_builder(email_client, with_improvements)


class TestEpisodicMDP(object):

    def test_validate(self, grid_world):
        P = grid_world.transitions
        R = grid_world.rewards
        gamma = grid_world.discount_factor
        S = grid_world.num_states
        S0 = grid_world.initial_state
        Sn = grid_world.terminal_state
        A = grid_world.num_actions
        invalid_P = np.random.rand(A, S, S)
        assert_raises(ValueError, EpisodicMDP, invalid_P, R, gamma, S0, Sn)
        invalid_R = np.random.rand(A, S, S)
        assert_raises(ValueError, EpisodicMDP, P, invalid_R, gamma, S0, Sn)
        invalid_gamma = 2
        assert_raises(ValueError, EpisodicMDP, P, R, invalid_gamma, S0, Sn)

    def test_solve(self, grid_world):
        P = grid_world.transitions
        R = grid_world.rewards
        gamma = grid_world.discount_factor
        mdp = EpisodicMDP(P, R, gamma)
        policy = mdp.solve()
        assert all([policy[i] == grid_world.policy[i]
                    for i in range(len(grid_world.policy))])

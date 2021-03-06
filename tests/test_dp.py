import numpy as np
from numpy.testing import assert_raises

from configurator.dialogs import Dialog
from configurator.dialogs.dp import DPDialogBuilder, EpisodicMDP


class TestDPDialog(object):

    def test_save(self, tmpdir, email_client):
        builder = DPDialogBuilder(email_client.var_domains,
                                  email_client.sample,
                                  rules=email_client.rules)
        saved_dialog = builder.build_dialog()
        file_path = str(tmpdir.join("dialog.zip"))
        saved_dialog.save(file_path)
        new_dialog = Dialog.load(file_path)
        assert new_dialog.var_domains == saved_dialog.var_domains
        assert new_dialog._policy == saved_dialog._policy
        new_dialog.reset()
        assert new_dialog.get_next_question() == 1
        new_dialog.set_answer(1, "lgi")
        assert new_dialog.config == {0: "no", 1: "lgi"}


class TestDPDialogBuilder(object):

    def test_build_dialog(self, email_client):
        builder = DPDialogBuilder(email_client.var_domains,
                                  email_client.sample,
                                  rules=email_client.rules,
                                  validate=True)
        dialog = builder.build_dialog()
        for config, num_questions in email_client.scenarios:
            dialog.reset()
            for i in range(num_questions):
                var_index = dialog.get_next_question()
                dialog.set_answer(var_index, config[var_index])
            assert dialog.is_complete()
            assert dialog.config == config


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

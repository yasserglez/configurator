import numpy as np
import mdptoolbox.example

from ..mdp import MDP, EpisodicMDP


class TestMDP(object):

    def setup(self):
        np.random.seed(42)
        self.S, self.A = 100, 5
        self.P, self.R = mdptoolbox.example.rand(self.S, self.A)
        self.gamma = 0.9

    def test_init_attributes(self):
        mdp = MDP(self.P, self.R, self.gamma, validate=False)
        np.testing.assert_array_equal(mdp.transitions, self.P)
        np.testing.assert_array_equal(mdp.rewards, self.R)
        assert mdp.discount_factor == self.gamma

    def test_init_validate(self):
        invalid_P = np.random.rand(self.A, self.S, self.S)
        np.testing.assert_raises(ValueError, MDP, invalid_P, self.R, self.gamma)
        invalid_R = np.zeros((self.A - 1, self.S, self.S))
        np.testing.assert_raises(ValueError, MDP, self.P, invalid_R, self.gamma)
        invalid_gamma = 2
        np.testing.assert_raises(ValueError, MDP, self.P, self.R, invalid_gamma)


class TestEpisodicMDP(object):

    def setup(self):
        pass

#    def test_init_attributes(self):
#        pass

#    def test_init_validate(self):
#        pass

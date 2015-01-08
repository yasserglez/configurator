import numpy as np
from numpy.testing import assert_array_equal, assert_raises

from .examples import load_gridworld
from ..mdp import MDP, EpisodicMDP, PolicyIteration, ValueIteration


S, A, P, R, INITIAL_S, TERMINAL_S, GAMMA, POLICY = load_gridworld()


class TestMDP(object):

    mdp_class = MDP

    def setup(self):
        np.random.seed(42)

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
        invalid_GAMMA = 2
        assert_raises(ValueError, self.mdp_class, P, R, invalid_GAMMA)


class TestEpisodicMDP(TestMDP):

    mdp_class = EpisodicMDP

    def test_init_attributes(self):
        mdp = self.mdp_class(P, R, GAMMA, INITIAL_S, TERMINAL_S)
        assert mdp.initial_state == INITIAL_S
        assert mdp.terminal_state == TERMINAL_S

    def test_init_validation(self):
        super().test_init_validation()
        invalid_P = np.random.rand(A, S, S)
        assert_raises(ValueError, self.mdp_class,
                      invalid_P, R, GAMMA, INITIAL_S, TERMINAL_S)
        invalid_R = np.random.rand(A, S, S)
        assert_raises(ValueError, self.mdp_class,
                      P, invalid_R, GAMMA, INITIAL_S, TERMINAL_S)


class BaseTestMDPSolver(object):

    solver_class = None

    def setup(self):
        self.solver = self.solver_class()

    def test_solve(self):
        mdp = EpisodicMDP(P, R, GAMMA)
        policy = self.solver.solve(mdp)
        assert policy == {s - 1: a for s, a in POLICY.items()}


class TestPolicyIteration(BaseTestMDPSolver):

    solver_class = PolicyIteration


class TestValueIteration(BaseTestMDPSolver):

    solver_class = ValueIteration

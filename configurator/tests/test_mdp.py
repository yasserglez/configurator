import numpy as np
from numpy.testing import assert_array_equal, assert_raises

from .examples import load_grid_world
from ..mdp import MDP, EpisodicMDP, PolicyIteration, ValueIteration


grid_world = load_grid_world()
S = grid_world.num_states
INITIAL_STATE = grid_world.initial_state
TERMINAL_STATE = grid_world.terminal_state
A = grid_world.num_actions
P = grid_world.transitions
R = grid_world.rewards
GAMMA = grid_world.discount_factor
POLICY = grid_world.policy
del grid_world


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
        invalid_gamma = 2
        assert_raises(ValueError, self.mdp_class, P, R, invalid_gamma)


class TestEpisodicMDP(TestMDP):

    mdp_class = EpisodicMDP

    def test_init_attributes(self):
        mdp = self.mdp_class(P, R, GAMMA, INITIAL_STATE, TERMINAL_STATE)
        assert mdp.initial_state == INITIAL_STATE
        assert mdp.terminal_state == TERMINAL_STATE

    def test_init_validation(self):
        super().test_init_validation()
        invalid_P = np.random.rand(A, S, S)
        assert_raises(ValueError, self.mdp_class,
                      invalid_P, R, GAMMA, INITIAL_STATE, TERMINAL_STATE)
        invalid_R = np.random.rand(A, S, S)
        assert_raises(ValueError, self.mdp_class,
                      P, invalid_R, GAMMA, INITIAL_STATE, TERMINAL_STATE)


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

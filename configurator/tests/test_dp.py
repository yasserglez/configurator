import numpy as np
from numpy.testing import assert_array_equal, assert_raises

from .examples import load_grid_world
from ..dp import MDP, EpisodicMDP, PolicyIteration, ValueIteration


grid_world = load_grid_world()
S, A = grid_world.num_states, grid_world.num_actions
S0, Sn = grid_world.initial_state, grid_world.terminal_state
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


class BaseTestMDPSolver(object):

    solver_class = None

    def setup(self):
        self.solver = self.solver_class()

    def test_solve(self):
        mdp = EpisodicMDP(P, R, GAMMA)
        policy = self.solver.solve(mdp)
        assert all([policy[i] == POLICY[i] for i in range(len(POLICY))])


class TestPolicyIteration(BaseTestMDPSolver):

    solver_class = PolicyIteration


class TestValueIteration(BaseTestMDPSolver):

    solver_class = ValueIteration

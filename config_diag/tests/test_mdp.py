import numpy as np

from ..mdp import MDP, EpisodicMDP, PolicyIteration, ValueIteration


# Example 4.1 of Reinforcement Learning: An Introduction
# by Richard S. Sutton and Andrew G. Barto.

GRIDWORLD_S = 15  # {1, 2, ..., 14} and the terminal state 15.
GRIDWORLD_INITIAL_S, GRIDWORLD_TERMINAL_S = 11, 14
GRIDWORLD_A = 4  # {up, down, right, left}.
up, down, right, left = range(GRIDWORLD_A)

# Transitions.
GRIDWORLD_P = np.zeros((GRIDWORLD_A, GRIDWORLD_S, GRIDWORLD_S))

# Grid transitions.
grid_transitions = {
    1: ((down, 5), (right, 2), (left, 15)),
    2: ((down, 6), (right, 3), (left, 1)),
    3: ((down, 7), (left, 2)),
    4: ((up, 15), (down, 8), (right, 5)),
    5: ((up, 1), (down, 9), (right, 6), (left, 4)),
    6: ((up, 2), (down, 10), (right, 7), (left, 5)),
    7: ((up, 3), (down, 11), (left, 6)),
    8: ((up, 4), (down, 12), (right, 9)),
    9: ((up, 5), (down, 13), (right, 10), (left, 8)),
    10: ((up, 6), (down, 14), (right, 11), (left, 9)),
    11: ((up, 7), (down, 15), (left, 10)),
    12: ((up, 8), (right, 13)),
    13: ((up, 9), (right, 14), (left, 12)),
    14: ((up, 10), (right, 15), (left, 13)),
}
for i, moves in grid_transitions.items():
    for a, j in moves:
        GRIDWORLD_P[a, i - 1, j - 1] = 1.0

# Border transitions.
for i in (1, 2, 3):
    GRIDWORLD_P[up, i - 1, i - 1] = 1.0
for i in (12, 13, 14):
    GRIDWORLD_P[down, i - 1, i - 1] = 1.0
for i in (3, 7, 11):
    GRIDWORLD_P[right, i - 1, i - 1] = 1.0
for i in (4, 8, 12):
    GRIDWORLD_P[left, i - 1, i - 1] = 1.0

# 15 should be an absorbing state.
GRIDWORLD_P[:, GRIDWORLD_TERMINAL_S, GRIDWORLD_TERMINAL_S] = 1.0

# Rewards.
GRIDWORLD_R = -1 * np.ones((GRIDWORLD_A, GRIDWORLD_S, GRIDWORLD_S))
GRIDWORLD_R[:, GRIDWORLD_TERMINAL_S, :] = 0

GRIDWORLD_GAMMA = 1.0

GRIDWORLD_POLICY = {
    1: left,
    2: left,
    3: down,
    4: up,
    5: up,
    6: up,
    7: down,
    8: up,
    9: up,
    10: down,
    11: down,
    12: up,
    13: right,
    14: right,
}

del up, down, right, left


class TestMDP(object):

    mdp_class = MDP

    def setup(self):
        np.random.seed(42)

    def test_init_attributes(self):
        mdp = self.mdp_class(GRIDWORLD_P, GRIDWORLD_R, GRIDWORLD_GAMMA)
        np.testing.assert_array_equal(mdp.transitions, GRIDWORLD_P)
        np.testing.assert_array_equal(mdp.rewards, GRIDWORLD_R)
        assert mdp.discount_factor == GRIDWORLD_GAMMA

    def test_init_validation(self):
        invalid_P = np.random.rand(GRIDWORLD_A, GRIDWORLD_S, GRIDWORLD_S)
        np.testing.assert_raises(ValueError, self.mdp_class,
                                 invalid_P, GRIDWORLD_R, GRIDWORLD_GAMMA)
        invalid_R = np.zeros((GRIDWORLD_A - 1, GRIDWORLD_S, GRIDWORLD_S))
        np.testing.assert_raises(ValueError, self.mdp_class,
                                 GRIDWORLD_P, invalid_R, GRIDWORLD_GAMMA)
        invalid_gamma = 2
        np.testing.assert_raises(ValueError, self.mdp_class,
                                 GRIDWORLD_P, GRIDWORLD_R, invalid_gamma)


class TestEpisodicMDP(TestMDP):

    mdp_class = EpisodicMDP

    def test_init_attributes(self):
        mdp = self.mdp_class(GRIDWORLD_P, GRIDWORLD_R, GRIDWORLD_GAMMA,
                             GRIDWORLD_INITIAL_S, GRIDWORLD_TERMINAL_S)
        assert mdp.initial_state == GRIDWORLD_INITIAL_S
        assert mdp.terminal_state == GRIDWORLD_TERMINAL_S

    def test_init_validation(self):
        super().test_init_validation()
        invalid_P = np.random.rand(GRIDWORLD_A, GRIDWORLD_S, GRIDWORLD_S)
        np.testing.assert_raises(ValueError, self.mdp_class,
                                 invalid_P, GRIDWORLD_R,
                                 GRIDWORLD_GAMMA, GRIDWORLD_INITIAL_S,
                                 GRIDWORLD_TERMINAL_S)
        invalid_R = np.random.rand(GRIDWORLD_A, GRIDWORLD_S, GRIDWORLD_S)
        np.testing.assert_raises(ValueError, self.mdp_class,
                                 GRIDWORLD_P, invalid_R,
                                 GRIDWORLD_GAMMA, GRIDWORLD_INITIAL_S,
                                 GRIDWORLD_TERMINAL_S)


class BaseTestMDPSolver(object):

    solver_class = None

    def setup(self):
        self.solver = self.solver_class()

    def test_solve(self):
        mdp = EpisodicMDP(GRIDWORLD_P, GRIDWORLD_R, GRIDWORLD_GAMMA)
        policy = self.solver.solve(mdp)
        assert policy == {s - 1: a for s, a in GRIDWORLD_POLICY.items()}


class TestPolicyIteration(BaseTestMDPSolver):

    solver_class = PolicyIteration


class TestValueIteration(BaseTestMDPSolver):

    solver_class = ValueIteration

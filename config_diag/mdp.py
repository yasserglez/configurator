"""
Markov Decision Processes
"""

import numpy.ma

from mdptoolbox.util import check
from mdptoolbox.mdp import PolicyIteration as PolicyIteration_
from mdptoolbox.mdp import ValueIteration as ValueIteration_


class MDP(object):
    """Finite MDP.

    Attributes:
        transitions: Transition probability matrices.
        rewards: Reward matrices.
        discount_factor: Discount factor in [0,1].
    """

    def __init__(self, transitions, rewards, discount_factor, validate=True):
        """Initialize a new instance.

        Let S denote the number of states and A the number of actions.

        Arguments:
            transitions: Transition probability matrices. Either a
                (A, S, S)-shaped numpy array or a list with A elements
                containing (possibly sparse) (S, S) matrices. An entry
                (a, s, s') gives the probability of transitioning from
                the state s into state s' with the action a.
            rewards: Reward matrices. Same shape as transitions.
                An entry (a, s, s') gives the reward for reaching the
                state s' from the state s by taking the action a.
            discount_factor: Discount factor in [0,1].
            validate: Set it to True (default) if the MDP formulation
                should be validated, False otherwise. A ValueError
                exception will be raised if any error is found.
        """
        super().__init__()
        self.transitions = transitions
        self.rewards = rewards
        self.discount_factor = discount_factor
        if validate:
            self._validate()

    def _validate(self):
        if not (0 <= self.discount_factor <= 1):
            raise ValueError("Discount factor not in [0,1]")
        # Rely on pymdptoolbox to check the transition and reward matrices.
        try:
            check(self.transitions, self.rewards)
        except AssertionError as error:
            raise ValueError(str(error))
        except mdptoolbox.util.InvalidMDPError as error:
            raise ValueError(error.message)


class EpisodicMDP(MDP):
    """Episodic MDP.

    Attributes:
        transitions: Transition probability matrices.
        rewards: Reward matrices.
        discount_factor: Discount factor in [0,1].
        initial_state: Index of the initial state.
        terminal_state: Index of the terminal state.
    """

    def __init__(self, transitions, rewards, discount_factor,
                 initial_state=None, terminal_state=None, validate=True):
        """Initialize a new instance.

        Let S denote the number of states and A the number of actions.

        Arguments:
            transitions: Transition probability matrices.
            rewards: Reward matrices.
            discount_factor: Discount factor in [0,1].
            initial_state: Index of the initial state (default: 0).
            terminal_state: Index of the terminal state (default: S - 1).
            validate: Set it to True (default) if the MDP formulation
                should be validated, False otherwise. A ValueError
                exception will be raised if any error is found.
        """
        self.initial_state = initial_state if initial_state is not None else 0
        self.terminal_state = (terminal_state
                               if terminal_state is not None else
                               transitions[0].shape[0] - 1)
        super().__init__(transitions, rewards, discount_factor, validate)

    def _validate(self):
        super()._validate()
        # Check that the terminal state is absorbing.
        s = self.terminal_state
        to_terminal = self.transitions[:, s, s]
        to_non_terminal = numpy.ma.array(self.transitions[:, s, :], mask=False)
        to_non_terminal.mask[:, s] = True
        if (to_terminal != 1).any() or to_non_terminal.any():
            raise ValueError("The terminal state is not an absorbing state")
        if self.rewards[:, s, s].any():
            raise ValueError("Terminal state has transitions with non-zero rewards")


class MDPSolver(object):
    """MDP solver.

    Attributes:
        mdp: MDP instance.
    """

    def __init__(self, mdp):
        """Initialize a new instance.

        Arguments:
            mdp: MDP instance.
        """
        super().__init__()
        self.mdp = mdp

    def solve(self):
        """Solve an MDP.

        Returns:
            A policy, i.e. a dict mapping state indexes to action indexes.
        """
        raise NotImplementedError()


class PolicyIteration(MDPSolver):
    """MDP solver using (modified) policy iterantion.

    Attributes:
        mdp: MDP instance.
    """

    def __init__(self, mdp):
        """Initialize a new instance.

        Arguments:
            mdp: MDP instance.
        """
        super().__init__(mdp)

    def solve(self, max_iter=1000):
        """Run the (modified) policy iteration algorithm.

        The initial policy is the one that maximizes the expected
        immediate rewards. The algorithm terminates when the policy
        does not change between two consecutive iterations or after a
        maximum number of iterations. The policy evaluation step is
        implemented iteratively, stopping when the maximum change in
        the value function is less than the threshold computed for
        epsilon 0.0001 (only for discounted MDPs) or after 10000
        iterations.

        Arguments:
            max_iter: The maximum number of iterations (default: 1000).

        Returns:
            A policy, i.e. a dict mapping state indexes to action indexes.
        """
        P = self.mdp.transitions
        R = self.mdp.rewards
        gamma = self.mdp.discount_factor
        pi = PolicyIteration_(P, R, gamma, max_iter=max_iter, eval_type="iterative")
        pi.run()
        policy = {s: a for s, a in enumerate(pi.policy)
                  if (not isinstance(self.mdp, EpisodicMDP) or
                      s != self.mdp.terminal_state)}
        return policy


class ValueIteration(MDPSolver):
    """MDP solver using value iteration.

    Attributes:
        mdp: MDP instance.
    """

    def __init__(self, mdp):
        """Initialize a new instance.

        Arguments:
            mdp: MDP instance.
        """
        super().__init__(mdp)

    def solve(self, epsilon=0.01, max_iter=1000):
        """Run the value iteration algorithm.

        The initial value function is zero for all the states. The
        algorithm terminates when an epsilon-optimal policy is found
        or after a maximum number of iterations.

        Parameters:
            epsilon: Stopping criterion (default: 0.01). For discounted
                MDPs it defines the epsilon value used to compute the
                threshold for the maximum change in the value function
                between two subsequent iterations. For undiscounted
                MDPs it defines the absolute threshold.
            max_iter: The maximum number of iterations (default: 1000).

        Returns:
            A policy, i.e. a dict mapping state indexes to action indexes.
        """
        P = self.mdp.transitions
        R = self.mdp.rewards
        gamma = self.mdp.discount_factor
        vi = ValueIteration_(P, R, gamma, epsilon, max_iter)
        vi.run()
        policy = {s: a for s, a in enumerate(vi.policy)
                  if (not isinstance(self.mdp, EpisodicMDP) or
                      s != self.mdp.terminal_state)}
        return policy

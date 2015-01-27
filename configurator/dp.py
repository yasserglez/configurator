"""Dynamic Programming"""

import os
import types
from contextlib import redirect_stdout

import mdptoolbox.util
from mdptoolbox.mdp import PolicyIteration as _PolicyIteration
from mdptoolbox.mdp import ValueIteration as _ValueIteration


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
            raise ValueError("The discount factor is not in [0,1]")
        # Rely on pymdptoolbox to check the transition and reward matrices.
        try:
            mdptoolbox.util.check(self.transitions, self.rewards)
        except AssertionError as error:
            raise ValueError(str(error)) from error
        except mdptoolbox.error.Error as error:
            raise ValueError(error.message) from error


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
        for a in range(len(self.transitions)):
            for sp in range(self.transitions[0].shape[0]):
                value = self.transitions[a][s, sp]
                if (s == sp and value != 1) or (s != sp and value != 0):
                    raise ValueError("The terminal state is not an absorbing state")
            if self.rewards[a][s, s] != 0:
                raise ValueError("Terminal state has transitions with non-zero rewards")


class MDPSolver(object):
    """MDP solver.
    """

    def __init__(self):
        """Initialize a new instance.
        """
        super().__init__()

    def solve(self, mdp):
        """Solve an MDP.

        Arguments:
            mdp: An MDP instance.

        Returns:
            A policy, i.e. a dict mapping state indices to action indices.
        """
        raise NotImplementedError()


class PolicyIteration(MDPSolver):
    """MDP solver using modified policy iterantion.

    The initial policy is the one that maximizes the expected
    immediate rewards. The algorithm terminates when the policy does
    not change between two consecutive iterations or after max_iter.
    The policy evaluation step is implemented iteratively, stopping
    when the maximum change in the value function is less than the
    threshold computed for eval_epsilon (only for discounted MDPs) or
    after eval_max_iter iterations.
    """

    def __init__(self, max_iter=1000, eval_epsilon=0.01, eval_max_iter=10):
        """Initialize a new instance.

        Arguments:
            max_iter: The maximum number of iterations (default: 1000).
            eval_epsilon: Stopping criterion for the policy evaluation
                step (default: 0.01). For discounted MDPs it defines
                the epsilon value used to compute the threshold for
                the maximum change in the value function between two
                subsequent iterations. It has no effect for
                undiscounted MDPs.
            eval_max_iter: Stopping criterion for the policy
                evaluation step (default: 10). The maximum number of
                iterations.
        """
        super().__init__()
        self._max_iter = max_iter
        self._eval_epsilon = eval_epsilon
        self._eval_max_iter = eval_max_iter

    def solve(self, mdp):
        """Run the modified policy iteration algorithm.

        Arguments:
            mdp: An MDP instance.

        Returns:
            A policy, i.e. a dict mapping state indices to action indices.
        """
        P = mdp.transitions
        R = mdp.rewards
        gamma = mdp.discount_factor
        # Prevent pymdptoolbox from printing a warning about using a
        # discount factor of 1.0.
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
            pi = _PolicyIteration(P, R, gamma, max_iter=self._max_iter,
                                  eval_type="iterative", skip_check=True)
        # Monkey-patch the _evalPoicyIterative method to change the
        # default values for epsilon and max_iter.
        original = pi._evalPolicyIterative
        epsilon = self._eval_epsilon
        max_iter = self._eval_max_iter
        patched = lambda self, V0=0, epsilon=epsilon, max_iter=max_iter: \
            original(V0, epsilon, max_iter)
        pi._evalPolicyIterative = types.MethodType(patched, pi)
        pi.run()
        policy = {s: a for s, a in enumerate(pi.policy)
                  if (not isinstance(mdp, EpisodicMDP) or
                      s != mdp.terminal_state)}
        return policy


class ValueIteration(MDPSolver):
    """MDP solver using value iteration.

    The initial value function is zero for all the states. The
    algorithm terminates when an epsilon-optimal policy is found or
    after a maximum number of iterations.
    """

    def __init__(self, epsilon=0.001, max_iter=1000):
        """Initialize a new instance.

        Parameters:
            epsilon: Stopping criterion (default: 0.001). For discounted
                MDPs it defines the epsilon value used to compute the
                threshold for the maximum change in the value function
                between two subsequent iterations. For undiscounted
                MDPs it defines the absolute threshold.
            max_iter: The maximum number of iterations (default: 1000).
        """
        super().__init__()
        self._epsilon = epsilon
        self._max_iter = max_iter

    def solve(self, mdp):
        """Run the value iteration algorithm.

        Arguments:
            mdp: An MDP instance.

        Returns:
            A policy, i.e. a dict mapping state indices to action indices.
        """
        P = mdp.transitions
        R = mdp.rewards
        gamma = mdp.discount_factor
        # Prevent pymdptoolbox from printing a warning about using a
        # discount factor of 1.0.
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
            vi = _ValueIteration(P, R, gamma, self._epsilon,
                                 self._max_iter, skip_check=True)
        vi.run()
        policy = {s: a for s, a in enumerate(vi.policy)
                  if (not isinstance(mdp, EpisodicMDP) or
                      s != mdp.terminal_state)}
        return policy

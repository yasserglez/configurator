"""
Markov Decision Processes
"""

import numpy.ma as ma
import mdptoolbox.util


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
            mdptoolbox.util.check(self.transitions, self.rewards)
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
        to_non_terminal = ma.array(self.transitions[:, s, :], mask=False)
        to_non_terminal.mask[:, s] = True
        if (to_terminal != 1).any() or to_non_terminal.any():
            raise ValueError("The terminal state is not an absorbing state")
        if self.rewards[:, s, s].any():
            raise ValueError("Terminal state has transitions with non-zero rewards")

"""
Markov Decision Processes
"""

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
            raise ValueError(error.message[len("PyMDPToolbox: "):])


class EpisodicMDP(MDP):
    """Undiscounted, episodic, finite MDP.

    Attributes:
        transitions: Transition probability matrices.
        rewards: Reward matrices.
        discount_factor: Set to 1.0, for an undiscounted MDP.
        initial_state: Index of the initial state.
        terminal_state: Index of the terminal state.
    """

    def __init__(self, transitions, rewards, initial_state,
                 terminal_state, validate=True):
        """Initialize a new instance.

        Arguments:
            transitions: Transition probability matrices.
            rewards: Reward matrices.
            initial_state: Index of the initial state.
            terminal_state: Index of the terminal state.
            validate: Set it to True (default) if the MDP formulation
                should be validated, False otherwise. A ValueError
                exception will be raised if any error is found.
        """
        super(EpisodicMDP, self).__init__(transitions, rewards, 1.0, validate)
        self.initial_state = initial_state
        self.terminal_state = terminal_state

    def _validate(self):
        super(EpisodicMDP, self)._validate()

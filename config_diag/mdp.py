"""
Markov Decision Processes
"""


class MDP(object):
    """Finite MDP.

    Attributes:
        transitions: Transition probability matrices.
        rewards: Reward matrices.
        discount_factor: Discount factor in [0,1].
    """

    def __init__(self, transitions, rewards, discount_factor):
        """Initialize a new instance.

        Let n denote the number of states and m the number of actions.

        Arguments:
            transitions: Transition probability matrices. Either a
                (m, n, n)-shaped numpy array or a list with m elements
                containing (possibly sparse) (n, n) matrices. An entry
                (a, s, s') gives the probability of transitioning from
                the state s into state s' with the action a.
            rewards: Reward matrices. Same shape as transitions.
                An entry (a, s, s') gives the reward for reaching the
                state s' from the state s by taking the action a.
            discount_factor: Discount factor in [0,1].
        """
        self.transitions = transitions
        self.rewards = rewards
        self.discount_factor = discount_factor


class EpisodicMDP(MDP):
    """Undiscounted, episodic, finite MDP.

    Attributes:
        transitions: Transition probability matrices.
        rewards: Reward matrices.
        discount_factor: Set to 1.0, for an undiscounted MDP.
        initial_state: Index of the initial state.
        terminal_state: Index of the terminal state.
    """

    def __init__(self, transitions, rewards, initial_state, terminal_state):
        """Initialize a new instance.

        Arguments:
            transitions: Transition probability matrices.
            rewards: Reward matrices.
            initial_state: Index of the initial state.
            terminal_state: Index of the terminal state.
        """
        super(EpisodicMDP, self).__init__(transitions, rewards, 1.0)
        self.initial_state = initial_state
        self.terminal_state = terminal_state

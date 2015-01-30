"""Configuration Dialogs based on Reinforcement Learning"""

import logging
import pprint
from functools import reduce
from operator import mul

import numpy as np
import numpy.ma as ma
from scipy import stats
from mdptoolbox.util import getSpan
from pybrain.rl.environments.environment import Environment
from pybrain.rl.environments.episodic import EpisodicTask
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import Q as Q_, SARSA as SARSA_
from pybrain.rl.learners.valuebased import ActionValueTable
from pybrain.rl.experiments import EpisodicExperiment

from .base import Dialog, DialogBuilder
from .util import iter_config_states


log = logging.getLogger(__name__)


class RLDialog(Dialog):
    """Configuration dialog generated using reinforcement learning.

    See Dialog for information about the attributes and methods.
    """

    def __init__(self, config_values, rules, table, validate=False):
        """Initialize a new instance.

        Arguments:
            table: A DialogQTable instance.

        See Dialog for the remaining arguments.
        """
        self._table = table
        super().__init__(config_values, rules, validate=validate)

    def get_next_question(self):
        """Get the question that should be asked next.
        """
        return self._table.get_next_question(self.config)


class RLDialogBuilder(DialogBuilder):
    """Build a configuration dialog using reinforcement learning.
    """

    def __init__(self, rl_algorithm="q-learning",
                 rl_table="exact",
                 rl_learning_rate=0.5,
                 rl_epsilon=0.3,
                 rl_epsilon_decay=1.0,
                 rl_max_episodes=1000,
                 **kwargs):
        """Initialize a new instance.

        Arguments:
            rl_algorithm: Reinforcement learning algorithm. Possible
                values are: 'q-learning' (default) and 'sarsa'.
            rl_learning_rate: Q-learning and SARSA learning rate
                (default: 0.5).
            rl_epsilon: Initial epsilon value for the epsilon-greedy
                exploration strategy (default: 0.3).
            rl_epsilon_decay: Epsilon decay rate (default: 1.0).
                The epsilon value is decayed after every episode.
            rl_max_episodes: Maximum number of simulated episodes
                (default: 1000).

        See DialogBuilder for the remaining arguments.
        """
        super().__init__(**kwargs)
        if rl_algorithm in ("q-learning", "sarsa"):
            self._rl_algorithm = rl_algorithm
        else:
            raise ValueError("Invalid rl_algorithm value")
        if rl_table in ("exact", "approx"):
            self._rl_table = rl_table
        else:
            raise ValueError("Invalid rl_table value")
        self._rl_learning_rate = rl_learning_rate
        self._rl_epsilon = rl_epsilon
        self._rl_epsilon_decay = rl_epsilon_decay
        self._rl_max_episodes = rl_max_episodes
        self._Vspan_threshold = 0.001

    def build_dialog(self):
        """Construct a configuration dialog.

        Returns:
            An instance of a Dialog subclass.
        """
        rules = self._mine_rules()
        dialog = Dialog(self._config_values, rules)
        env = DialogEnvironment(dialog, self._freq_tab)
        task = DialogTask(env)
        if self._rl_table == "exact":
            table = DialogQTable(self._config_values)
        elif self._rl_table == "approx":
            table = ApproxDialogQTable(self._config_values)
        if self._rl_algorithm == "q-learning":
            learner = QLearning(alpha=self._rl_learning_rate, gamma=1.0)
        elif self._rl_algorithm == "sarsa":
            learner = SARSA(alpha=self._rl_learning_rate, gamma=1.0)
        agent = DialogAgent(table, learner, self._rl_epsilon,
                            self._rl_epsilon_decay)
        exp = EpisodicExperiment(task, agent)
        log.info("running the RL algorithm")
        Vprev = table.Q.max(1)
        for curr_episode in range(self._rl_max_episodes):
            exp.doEpisodes(number=1)
            agent.learn(episodes=1)
            agent.reset()
            # Check the stopping criterion.
            V = table.Q.max(1)
            Verror = getSpan(V - Vprev)
            Vprev = V
            if Verror < self._Vspan_threshold:
                break
        log.info("terminated after %d episodes", curr_episode + 1)
        log.info("finished running the RL algorithm")
        # Create the RLDialog instance.
        dialog = RLDialog(self._config_values, rules, table,
                          validate=self._validate)
        return dialog


class DialogQTable(ActionValueTable):
    """An action-value table."""

    def __init__(self, config_values):
        """Initialize a new instance.

        Arguments:
            config_values: A list with one entry for each variable,
                containing an enumerable with all the possible values
                of the variable.
        """
        self._config_values = config_values
        num_states = self._get_num_states()
        num_actions = len(self._config_values)
        log.info("the action-value table has %d states and %d actions",
                 num_states, num_actions)
        super().__init__(num_states, num_actions)
        self.initialize(-1)  # pessimistic initialization
        self.Q = self.params.reshape(num_states, num_actions)

    def _get_num_states(self):
        # One variable value is added for the unknown value.
        all_states = reduce(mul, map(lambda x: len(x) + 1, self._config_values))
        terminal_states = reduce(mul, map(len, self._config_values))
        num_states = (all_states - terminal_states) + 1
        return num_states

    def get_state_index(self, config):
        # Find the position of the state among all the possible
        # configuration states. This is not efficient, but the
        # tabular version won't work for many variables anyway.
        state_key = hash(frozenset(config.items()))
        non_terminals = iter_config_states(self._config_values, True)
        for state_index, state in enumerate(non_terminals):
            if state_key == hash(frozenset(state.items())):
                return state_index

    def get_next_question(self, config):
        # Get the question that should be asked at the given state.
        state = self.get_state_index(config)
        invalid_actions = [a in config for a in range(self.numActions)]
        action = ma.array(self.Q[state, :], mask=invalid_actions).argmax()
        return action

    def getMaxAction(self, state):
        # superclass method uses random.choice to choose among
        # multiple actions with the same maximum value, we just want
        # to return the first one.
        action = self.Q[state, :].argmax()
        return action


class ApproxDialogQTable(DialogQTable):
    """Approxximate action-value table using state aggregation."""

    def __init__(self, config_values):
        """Initialize a new instance.

        See DialogQTable for information about the arguments.
        """
        super().__init__(config_values)

    def _get_num_states(self):
        # All the states in which the same number of variables are
        # known are aggregated into a single state.
        num_states = len(self._config_values) + 1
        return num_states

    def get_state_index(self, config):
        state_index = len(config)
        log.debug("the aggregated state index is %d", state_index)
        return state_index


class DialogEnvironment(Environment):
    """Represents the use of a configuration dialog in PyBrain's RL model.

    The environment keeps track of the configuration state. It starts
    in a state where all the variables are unknown (and it can be
    reset at any time to this state using the reset method). An action
    can be performed (i.e. asking a question) by calling
    performAction. Then, the user response is simulated and the
    configuration state is updated (including variables discovered due
    to the association rules). The current configuration state is
    returned by getSensors.

    Attributes:
        dialog: A Dialog instance.
    """

    def __init__(self, dialog, freq_tab):
        """Initialize a new instance.

        Arguments:
            dialog: A Dialog instance.
            freq_tab: A FrequencyTable instance.
        """
        super().__init__()
        self.dialog = dialog
        self._freq_tab = freq_tab

    def reset(self):
        log.debug("configuration reset in the environment")
        self.dialog.reset()

    def getSensors(self):
        log.debug("current configuration in the environment:\n%s",
                  pprint.pformat(self.dialog.config))
        return self.dialog.config

    def performAction(self, action):
        log.debug("performing action %d in the environment", action)
        var_index = int(action)
        # Simulate the user response.
        values = ([], [])
        for var_value in self._freq_tab.var_values[var_index]:
            response = {var_index: var_value}
            var_prob = self._freq_tab.cond_prob(response, self.dialog.config)
            values[0].append(var_value)
            values[1].append(var_prob)
        var_value = stats.rv_discrete(values=values).rvs()
        log.debug("simulated user response %d", var_value)
        self.dialog.set_answer(var_index, var_value)
        log.debug("new configuration in the environment:\n%s",
                  pprint.pformat(self.dialog.config))
        log.debug("finished performing action %d in the environment", action)


class DialogTask(EpisodicTask):
    """Represents the configuration goal in PyBrain's RL model.
    """

    def __init__(self, env):
        """Initialize a new instance.

        Arguments:
            env: A DialogEnvironment instance.
        """
        super().__init__(env)
        self.lastreward = None

    def reset(self):
        super().reset()
        self.lastreward = None
        log.debug("the task was reset")

    def getObservation(self):
        log.debug("computing the observation in the task")
        config = self.env.getSensors()
        return config

    def performAction(self, action):
        log.debug("performing action %d in the task", action)
        config_len_before = len(self.env.dialog.config)
        self.env.performAction(action)
        config_len_after = len(self.env.dialog.config)
        # The reward is the number of variables that were set minus
        # the cost of asking the question.
        self.lastreward = (config_len_after - config_len_before) - 1
        self.cumreward += self.lastreward
        self.samples += 1
        log.debug("the computed reward is %d", self.lastreward)
        log.debug("finished performing action %d in the task", action)

    def addReward(self):
        # The reward is added in performAction, overriding and raising
        # an exception to make sure this method isn't called by PyBrain.
        raise NotImplementedError

    def getReward(self):
        log.debug("the reward for the last action is %d", self.lastreward)
        return self.lastreward

    def isFinished(self):
        is_finished = self.env.dialog.is_complete()
        if is_finished:
            log.debug("an episode finished, total reward %d", self.cumreward)
        return is_finished


class DialogAgent(LearningAgent):

    def __init__(self, table, learner, epsilon, epsilon_decay):
        """Initialize a new instance.

        Arguments:

            table: A DialogQTable instance.
            learner: A Q or SARSA instance.
            epsilon: Initial epsilon value for the epsilon-greedy
                exploration strategy.
            epsilon_decay: Epsilon decay rate. The epsilon value is
                decayed after every episode.
        """
        super().__init__(table, learner)  # self.module = table
        self._epsilon = epsilon * (1 / epsilon_decay)
        self._epsilon_decay = epsilon_decay

    def newEpisode(self):
        log.debug("new episode in the agent")
        super().newEpisode()
        self._epsilon *= self._epsilon_decay
        log.debug("the epsilon value is %g", self._epsilon)

    def integrateObservation(self, obs):
        self.lastconfig = obs
        state_index = self.module.get_state_index(self.lastconfig)
        super().integrateObservation(state_index)  # self.lastobs = state_index

    def getAction(self):
        log.debug("getting an action from the agent")
        self.lastaction = self.module.get_next_question(self.lastconfig)
        log.debug("the greedy action is %d", self.lastaction)
        # Epsilon-greedy exploration, i.e. check if the greedy action
        # should be replaced by a randomly chosen action. Not using
        # EpsilonGreedyExplorer because we want to restrict the
        # sampling to the valid actions.
        if np.random.uniform() < self._epsilon:
            invalid_actions = [i for i in range(self.module.numActions)
                               if i not in self.lastconfig]
            self.lastaction = np.random.choice(invalid_actions)
            log.debug("performing random action %d instead", self.lastaction)
        return self.lastaction


class _LearnFromLastMixin(object):

    def learn(self):
        # We need to process the reward for entering the terminal state
        # but Q.learn and SARSA.learn don't do it. Let Q and SARSA process
        # the complete episode first, and then process the last observation.
        # We assume Q is zero for the terminal state.
        super().learn()
        # This will only work if episodes are processed one by
        # one, so ensure there's only one sequence in the dataset.
        assert self.batchMode and self.dataset.getNumSequences() == 1
        seq = next(iter(self.dataset))
        for laststate, lastaction, lastreward in seq:
            pass  # skip all the way to the last
        laststate = int(laststate)
        lastaction = int(lastaction)
        qvalue = self.module.getValue(laststate, lastaction)
        new_qvalue = qvalue + self.alpha * (lastreward - qvalue)
        self.module.updateValue(laststate, lastaction, new_qvalue)


class QLearning(_LearnFromLastMixin, Q_):
    """Q-Learning."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SARSA(_LearnFromLastMixin, SARSA_):
    """SARSA."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

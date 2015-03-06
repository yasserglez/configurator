"""Configuration dialogs based on reinforcement learning.
"""

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

from .dialogs import Dialog, DialogBuilder
from .util import iter_config_states


__all__ = ["RLDialogBuilder"]


log = logging.getLogger(__name__)


class RLDialogBuilder(DialogBuilder):
    """Build a configuration dialog using reinforcement learning.

    Arguments:
        rl_algorithm: The reinforcement learning algorithm. Possible
            values are: `'q-learning'` and `'sarsa'`.
        rl_table: Representation of the action-value table. Possible
            values are `'exact'` (full table) and `'approximate'`
            (approximate table using state aggregation).
        rl_consistency: Type of consistency check used to filter the
            domain of the remaining questions during the simulation of
            the RL episodes. Possible values are: `'global'` and
            `'local'` (only implemented for binary constraints). This
            argument is ignored for rule-based dialogs.
        rl_learning_rate: Q-learning and SARSA learning rate.
        rl_epsilon: Initial epsilon value for the epsilon-greedy
            exploration strategy.
        rl_epsilon_decay: Epsilon decay rate. The epsilon value is
            decayed after every episode.
        rl_max_episodes: Maximum number of simulated episodes.

    See :class:`configurator.dialogs.DialogBuilder` for the remaining
    arguments.
    """

    def __init__(self, var_domains, rules=None, constraints=None, sample=None,
                 rl_algorithm="q-learning",
                 rl_table="approximate",
                 rl_consistency="local",
                 rl_learning_rate=0.3,
                 rl_epsilon=0.5,
                 rl_epsilon_decay=0.99,
                 rl_max_episodes=1000,
                 validate=False):
        super().__init__(var_domains, rules, constraints, sample, validate)
        if rl_algorithm in {"q-learning", "sarsa"}:
            self._rl_algorithm = rl_algorithm
        else:
            raise ValueError("Invalid rl_algorithm value")
        if rl_table in {"exact", "approximate"}:
            self._rl_table = rl_table
        else:
            raise ValueError("Invalid rl_table value")
        if rl_consistency in {"global", "local"}:
            self._rl_consistency = rl_consistency
        else:
            raise ValueError("Invalid rl_consistency value")
        self._rl_learning_rate = rl_learning_rate
        self._rl_epsilon = rl_epsilon
        self._rl_epsilon_decay = rl_epsilon_decay
        self._rl_max_episodes = rl_max_episodes
        self._Vspan_threshold = 0.01

    def build_dialog(self):
        """Construct a configuration dialog.

        Returns:
            An instance of a `configurator.dialogs.Dialog` subclass.
        """
        dialog = Dialog(self.var_domains, self.rules, self.constraints)
        env = DialogEnvironment(dialog, self._rl_consistency, self._freq_table)
        task = DialogTask(env)
        if self._rl_algorithm == "q-learning":
            learner = QLearning(alpha=self._rl_learning_rate, gamma=1.0)
        elif self._rl_algorithm == "sarsa":
            learner = SARSA(alpha=self._rl_learning_rate, gamma=1.0)
        if self._rl_table == "exact":
            table = DialogQTable(self.var_domains)
        elif self._rl_table == "approximate":
            table = ApproxDialogQTable(self.var_domains)
        agent = DialogAgent(table, learner, self._rl_epsilon,
                            self._rl_epsilon_decay)
        exp = EpisodicExperiment(task, agent)
        log.info("running the RL algorithm")
        Vprev = table.Q.max(1)
        complete_episodes = 0
        for curr_episode in range(self._rl_max_episodes):
            exp.doEpisodes(number=1)
            if env.dialog.is_complete():
                complete_episodes += 1
                agent.learn(episodes=1)
            agent.reset()
            # Check the stopping criterion.
            V = table.Q.max(1)
            Verror = getSpan(V - Vprev)
            Vprev = V
            if Verror < self._Vspan_threshold:
                break
        log.info("terminated after %d episodes", curr_episode + 1)
        log.info("learned from %d episodes", complete_episodes)
        log.info("finished running the RL algorithm")
        # Create the RLDialog instance.
        dialog = RLDialog(self.var_domains, table, self.rules,
                          self.constraints, validate=self._validate)
        return dialog


class RLDialog(Dialog):
    """Configuration dialog generated using reinforcement learning.

    Arguments:
        table: A `DialogQTable` instance.

    See `configurator.dialogs.Dialog` for information about the
    remaining arguments, attributes and methods.
    """

    def __init__(self, var_domains, table,
                 rules=None, constraints=None, validate=False):
        self._table = table
        super().__init__(var_domains, rules, constraints, validate)

    def get_next_question(self):
        next_question = self._table.get_next_question(self.config)
        return next_question


class DialogQTable(ActionValueTable):

    def __init__(self, var_domains):
        self.var_domains = var_domains
        self._var_card = list(map(len, self.var_domains))
        num_states = self._get_num_states()
        num_actions = len(self.var_domains)
        log.info("the action-value table has %d states and %d actions",
                 num_states, num_actions)
        super().__init__(num_states, num_actions)
        self.initialize(0)
        self.Q = self.params.reshape(num_states, num_actions)
        self.Q[num_states - 1, :] = 0

    def _get_num_states(self):
        # One variable value is added for the unknown state.
        all_states = reduce(mul, map(lambda x: x + 1, self._var_card))
        terminal_states = reduce(mul, map(len, self.var_domains))
        num_states = (all_states - terminal_states) + 1
        return num_states

    def get_state_index(self, config):
        # Find the position of the state among all the possible
        # configuration states. This implementation is not efficient,
        # but the exact version won't work for many variables anyway.
        state_key = hash(frozenset(config.items()))
        non_terminals = iter_config_states(self.var_domains, True)
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
        # The superclass method uses random.choice to choose among
        # multiple actions with the same maximum value, we just want
        # to return the first one.
        action = self.Q[state, :].argmax()
        return action


class ApproxDialogQTable(DialogQTable):

    def __init__(self, var_domains):
        super().__init__(var_domains)

    def _get_num_states(self):
        # Number of known variables. One added for the initial state
        # in which all the variables are unknown.
        num_states = len(self.var_domains) + 1
        return num_states

    def get_state_index(self, config):
        state_index = len(config)
        log.debug("the aggregated state index is %d", state_index)
        return state_index


class DialogEnvironment(Environment):
    """Represents a configuration dialog in PyBrain's RL model.

    Arguments:
        dialog: A :class:`configurator.dialogs.Dialog` instance.
        consistency: Whether to enforce `'global'` or `'local'`
            consistency after each answer.
        freq_table: A :class:`configurator.freq_table.FrequencyTable` instance.

    The environment keeps track of the configuration state. It starts
    in a state where all the variables are unknown (and it can be
    reset at any time to this state using the reset method). An action
    can be performed (i.e. asking a question) by calling
    :meth:`performAction`. Then, the user response is simulated and
    the configuration state is updated (including discovered
    variables). The current configuration state is returned by
    :meth:`getSensors`.

    Attributes:
        dialog: A :class:`configurator.dialogs.Dialog` instance.
    """

    def __init__(self, dialog, consistency, freq_table):
        super().__init__()
        self.dialog = dialog
        self._consistency = consistency
        self._freq_table = freq_table

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
        var_values = self.dialog.get_possible_answers(var_index)
        # Simulate the user response.
        values = ([], [])
        for i, var_value in enumerate(var_values):
            response = {var_index: var_value}
            prob = self._freq_table.cond_prob(response, self.dialog.config)
            if var_value is not None and prob > 0:
                values[0].append(i)
                values[1].append(prob)
        var_value = var_values[stats.rv_discrete(values=values).rvs()]
        log.debug("simulated user response %r", var_value)
        self.dialog.set_answer(var_index, var_value, self._consistency)
        log.debug("new configuration in the environment:\n%s",
                  pprint.pformat(self.dialog.config))
        log.debug("finished performing action %d in the environment", action)


class DialogTask(EpisodicTask):
    """Represents the configuration goal in PyBrain's RL model.

    Arguments:
        env: A `DialogEnvironment` instance.
    """

    def __init__(self, env):
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
        is_finished = False
        if self.env.dialog.is_complete():
            log.info("finished an episode, total reward %d", self.cumreward)
            is_finished = True
        elif not self.env.dialog.is_consistent():
            log.info("finished an episode, reached an inconsistent state")
            is_finished = True
        return is_finished


class DialogAgent(LearningAgent):
    """A learning agent in PyBrain's RL model.

    Arguments:
        table: A `DialogQTable` instance.
        learner: A `Q` or `SARSA` instance.
        epsilon: Initial epsilon value for the epsilon-greedy
           exploration strategy.
        epsilon_decay: Epsilon decay rate. The epsilon value is
            decayed after every episode.
    """

    def __init__(self, table, learner, epsilon, epsilon_decay):
        super().__init__(table, learner)  # self.module = table
        self._epsilon = epsilon * (1 / epsilon_decay)  # it's decreased first
        self._epsilon_decay = epsilon_decay

    def newEpisode(self):
        log.debug("new episode in the agent")
        super().newEpisode()
        self.lastconfig = None
        self.lastobs = None
        self.lastaction = None
        self.lastreward = None
        self._epsilon *= self._epsilon_decay
        log.info("the epsilon value is %g", self._epsilon)

    def integrateObservation(self, obs):  # Step 1
        self.lastconfig = obs
        state = self.module.get_state_index(self.lastconfig)
        self.lastobs = state

    def getAction(self):  # Step 2
        log.debug("getting an action from the agent")
        # Epsilon-greedy exploration. It ensures that a valid action
        # at the current configuration state is always returned
        # (i.e. a question that hasn't been answered).
        action = self.module.get_next_question(self.lastconfig)
        log.debug("the greedy action is %d", action)
        if np.random.uniform() < self._epsilon:
            valid_actions = [a for a in range(self.module.numActions)
                             if a not in self.lastconfig]
            action = np.random.choice(valid_actions)
            log.debug("performing random action %d instead", action)
        self.lastaction = action
        return self.lastaction

    def giveReward(self, reward):  # Step 3
        self.lastreward = reward
        self.history.addSample(self.lastobs, self.lastaction, self.lastreward)


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SARSA(_LearnFromLastMixin, SARSA_):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

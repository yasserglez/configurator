"""Reinforcement Learning"""

import logging
import pprint
from functools import reduce
from operator import mul

import numpy as np
from scipy import stats
from pybrain.rl.environments.environment import Environment
from pybrain.rl.environments.episodic import EpisodicTask
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import Q as Q_, SARSA as SARSA_
from pybrain.rl.learners.valuebased import ActionValueTable  # noqa
from pybrain.rl.experiments import EpisodicExperiment  # noqa

from .util import iter_config_states


log = logging.getLogger(__name__)


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
        num_states: The number of configuration states. All the
            terminal states (i.e. where all the variables are known)
            are represented by a single state.
        num_actions: The number of actions, i.e. the number of variables.
    """

    def __init__(self, freq_tab, rules):
        """Initialize a new instance.

        Arguments:
            freq_tab: A FrequencyTable instance.
            rules: A list of AssociationRule instances.
        """
        super().__init__()
        self._freq_tab = freq_tab
        self._rules = rules
        self.config_values = freq_tab.var_values
        # One is added for the unknown value.
        all_states = reduce(mul, map(lambda x: len(x) + 1, self.config_values))
        terminal_states = reduce(mul, map(len, self.config_values))
        self.num_states = all_states - terminal_states + 1
        self.num_actions = len(self.config_values)
        log.debug("the environment has %d states and %d actions",
                  self.num_states, self.num_actions)

    def reset(self):
        log.debug("the configuration was reset in the environment")
        self.config = {}

    def getSensors(self):
        log.debug("configuration in the environment:\n%s",
                  pprint.pformat(self.config))
        return self.config

    def performAction(self, action):
        log.debug("performing action %d in the environment", action)
        var_index = int(action)
        if var_index in self.config:
            raise ValueError("variable {0} is already known".format(var_index))
        else:
            # Simulate the user response.
            values = ([], [])
            for var_value in self.config_values[var_index]:
                response = {var_index: var_value}
                var_prob = self._freq_tab.cond_prob(response, self.config)
                values[0].append(var_value)
                values[1].append(var_prob)
            var_value = stats.rv_discrete(values=values).rvs()
            log.debug("simulated user response %d", var_value)
            # Update the configuration state with the new variable and
            # apply the association rules.
            self.config[var_index] = var_value
            for rule in self._rules:
                if rule.is_applicable(self.config):
                    rule.apply_rule(self.config)
            log.debug("the new configuration is:\n%s",
                      pprint.pformat(self.config))
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
        state = self.env.getSensors()
        # Compute the state index.
        if len(state) == len(self.env.config_values):
            # The collapsed terminal state has the last index.
            state_index = self.env.num_states - 1
        else:
            # Find the position of the state amongst all the possible
            # configuration states. This is not efficient, but the
            # tabular version won't work for many variables anyway.
            state_key = hash(frozenset(state.items()))
            non_terminals = iter_config_states(self.env.config_values, True)
            for i, state in enumerate(non_terminals):
                if state_key == hash(frozenset(state.items())):
                    state_index = i
                    break
        obs = (state_index, state)
        log.debug("observation in the task:\n%s", pprint.pformat(obs))
        return obs

    def performAction(self, action):
        log.debug("performing action %d in the task", action)
        num_known_vars_before = len(self.env.config)
        self.env.performAction(action)
        num_known_vars_after = len(self.env.config)
        # The reward is the number of variables that were set minus
        # the cost of asking the question.
        self.lastreward = (num_known_vars_after - num_known_vars_before) - 1
        self.cumreward += self.lastreward
        self.samples += 1
        log.debug("the computed reward is %d", self.lastreward)
        log.debug("finished performing action %d in the task", action)

    def addReward(self):
        # The reward is added in performAction, overriding and raising
        # an exception to make sure this method isn't called by PyBrain.
        raise NotImplementedError()

    def getReward(self):
        log.debug("the reward for the last action is %d", self.lastreward)
        return self.lastreward

    def isFinished(self):
        is_finished = len(self.env.config) == len(self.env.config_values)
        if is_finished:
            log.debug("the episode has finished (total reward %d)",
                      self.cumreward)
        return is_finished


class DialogLearningAgent(LearningAgent):

    def __init__(self, table, learner, epsilon):
        super().__init__(table, learner)
        self.epsilon = epsilon

    def integrateObservation(self, obs):
        state_index, state = obs
        self.lastindex = np.array([state_index])
        self.laststate = state
        super().integrateObservation(self.lastindex)

    def getAction(self):
        num_actions = self.module.numActions
        # self.module.activate returns the greedy action.
        self.lastaction = int(self.module.activate(self.lastobs))
        if self.lastaction in self.laststate:
            # Pick the first valid action if ActionValueTable returned
            # an invalid action (ActionValueTable.getMaxAction chooses
            # randomly when two or more actions have the same Q value).
            self.lastaction = next(iter((i for i in range(num_actions)
                                         if i not in self.laststate)))
        # Epsilon-greedy exploration. Check if the greedy action
        # should be replaced by a randomly chosen action. Didn't
        # use EpsilonGreedyExplorer because I couldn't find a way
        # to easily restrict the sampling to the valid actions.
        if np.random.uniform() < self.epsilon:
            valid_actions = [i for i in range(num_actions)
                             if i not in self.laststate]
            self.lastaction = np.random.choice(valid_actions)
        return self.lastaction


class _LearnFromLastMixin(object):

    def learn(self):
        # We need to process the reward for entering the terminal state
        # but Q.learn and SARSA.learn don't do it. Let Q and SARSA process
        # the complete episode first, and then process the last observation.
        # We assume Q is zero for the terminal state.
        super().learn()
        if self.batchMode:
            # This will only work if episodes are processed one by
            # one, so ensure there's only one sequence in the dataset.
            assert self.dataset.getNumSequences() == 1
            seq = next(iter(self.dataset))
            for laststate, lastaction, lastreward in seq:
                pass  # skip all the way to the last
            laststate = int(laststate)
            lastaction = int(lastaction)
            qvalue = self.module.getValue(laststate, lastaction)
            new_qvalue = qvalue + self.alpha * (lastreward - qvalue)
            self.module.updateValue(laststate, lastaction, new_qvalue)


class Q(_LearnFromLastMixin, Q_):
    """Q-Learning."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SARSA(_LearnFromLastMixin, SARSA_):
    """SARSA."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

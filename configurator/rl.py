"""Reinforcement Learning"""

import logging
import pprint

from scipy import stats
from pybrain.rl.environments.environment import Environment
from pybrain.rl.environments.episodic import EpisodicTask


log = logging.getLogger(__name__)


class ConfigDiagEnvironment(Environment):
    """Represents the use of a configuration dialog in PyBrain's RL model.

    The environment keeps track of the configuration state. It starts
    in a state where all the variables are unknown (and it can be
    reset at any time to this state using the reset method). An action
    can be performed (i.e. asking a question) by calling
    performAction. Then, the user response is simulated and the
    configuration state is updated (including variables discovered due
    to the association rules). The current configuration state is
    returned by getSensors.
    """

    def __init__(self, config_values, freq_tab, rules):
        """Initialize a new instance.

        Arguments:
            config_values: A list with one entry for each variable,
                containing an enumerable with all the possible values of
                the variable.
            freq_tab: A FrequencyTable instance.
            rules: A list of AssociationRule instances.
        """
        super().__init__()
        self.config_values = config_values
        self._freq_tab = freq_tab
        self._rules = rules
        self.reset()

    def reset(self):
        log.debug("configuration state reset")
        self.config = {}

    def getSensors(self):
        log.debug("returning the configuration state in the environment:\n%s",
                  pprint.pformat(self.config, width=60))
        return self.config

    def performAction(self, action):
        log.debug("performing action %d in the environment", action)
        # Simulate a user response.
        var_index = action
        values = []
        for var_value in self.config_values[var_index]:
            response = {var_index: var_value}
            var_prob = self._freq_tab.cond_prob(response, self.config)
            values.append((var_value, var_prob))
        var_value = stats.rv_discrete(values=values).rvs()
        log.debug("simulated user response %d", var_value)
        # Update the configuration state with the new variable and
        # apply the association rules.
        self.config[var_index] = var_value
        for rule in self._rules:
            if rule.is_applicable(self.config):
                rule.apply_rule(self.config)
        log.debug("the new configuration state is:\n%s",
                  pprint.pformat(self.config, width=60))


class ConfigDiagTask(EpisodicTask):
    """Represents the configuration goal in PyBrain's RL model.
    """

    def __init__(self, env):
        """Initialize a new instance.

        Arguments:
            env: A ConfigDiagEnvironment instance.
        """
        super().__init__(env)
        self._last_action_reward = None

    def reset(self):
        super().reset()
        self._last_action_reward = None
        log.debug("the task was reset")

    def getObservation(self):
        # TODO: What's the expected return value?
        config = self.env.getSensors()
        log.debug("returning the observation in the task")
        return config

    def performAction(self, action):
        log.debug("performing action %d in the task", action)
        var_index = action
        if var_index in self.env.config:
            log.debug("the question %d has already been answered", var_index)
            # Asked one question but acquired no new information, so
            # the reward is -1. This call isn't passed to the
            # environment because it would sample a new answer for the
            # question, moving the agent to a different state.
            self._last_action_reward = -1
        else:
            vars_known_before = len(self.env.config)
            self.env.performAction(action)
            vars_known_after = len(self.env.config)
            # The reward is the number of automatically discovered variables.
            self._last_action_reward = (vars_known_after -
                                        vars_known_before) - 1

    def getReward(self):
        log.debug("the reward for the last action is %d",
                  self._last_action_reward)
        return self._last_action_reward

    def isFinished(self):
        is_finished = len(self.env.config) == len(self.env.config_values)
        if is_finished:
            log.debug("the current episode has finished")
        return is_finished

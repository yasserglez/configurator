"""Configuration Dialogs based on Reinforcement Learning"""

import logging
import pprint
from functools import reduce
from operator import mul

import numpy as np
from scipy import stats
from mdptoolbox.util import getSpan
from pybrain.rl.environments.environment import Environment
from pybrain.rl.environments.episodic import EpisodicTask
from pybrain.rl.agents import LearningAgent
from pybrain.rl.learners import Q as Q_, SARSA as SARSA_
from pybrain.rl.learners.valuebased import ActionValueTable  # noqa
from pybrain.rl.experiments import EpisodicExperiment  # noqa

from .base import Dialog, DialogBuilder
from .util import iter_config_states


log = logging.getLogger(__name__)


class RLDialog(Dialog):
    """Configuration dialog generated using reinforcement learning.

    See Dialog for information about the attributes and methods.
    """

    def __init__(self, config_values, rules, policy, validate=False):
        """Initialize a new instance.

        Arguments:
            policy: The MDP policy, i.e. a dict mapping configuration
                states to variable indices. The configuration states
                are represented as frozensets of (index, value) tuples
                for each variable.

        See Dialog for the remaining arguments.
        """
        self._policy = policy
        super().__init__(config_values, rules, validate=validate)

    def _validate(self):
        for state in iter_config_states(self.config_values, True):
            state_key = frozenset(state.items())
            try:
                if self._policy[state_key] in state:
                    raise ValueError("The policy has invalid actions")
            except KeyError:
                # States that can be skipped using the association
                # rules won't appear on the policy.
                pass

    def get_next_question(self):
        """Get the question that should be asked next.
        """
        next_var_index = self._policy[frozenset(self.config.items())]
        return next_var_index


class RLDialogBuilder(DialogBuilder):
    """Build a configuration dialog using reinforcement learning.
    """

    def __init__(self, rl_algorithm="q-learning",
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
        log.debug("running the RL algorithm")
        env = DialogEnvironment(self._freq_tab, rules)
        task = DialogTask(env)
        table = ActionValueTable(env.num_states, env.num_actions)
        # Can't be initilized to a value greater than zero without
        # changing PyBrain's internals. ActionValueTable chooses
        # randomly among actions with the same Q value and it would
        # choose many invalid actions.
        table.initialize(-1)
        if self._rl_algorithm == "q-learning":
            learner = Q(alpha=self._rl_learning_rate, gamma=1.0)
        elif self._rl_algorithm == "sarsa":
            learner = SARSA(alpha=self._rl_learning_rate, gamma=1.0)
        agent = DialogLearningAgent(table, learner, self._rl_epsilon)
        exp = EpisodicExperiment(task, agent)
        Qvalues = table.params.reshape(env.num_states, env.num_actions)
        Vprev = Qvalues.max(1)
        for curr_episode in range(self._rl_max_episodes):
            log.debug("the epsilon value is %.2f", agent.epsilon)
            exp.doEpisodes(number=1)
            agent.learn(episodes=1)
            agent.reset()
            agent.epsilon *= self._rl_epsilon_decay
            # Check the stopping criterion.
            V = Qvalues.max(1)
            Verror = getSpan(V - Vprev)
            Vprev = V
            if Verror < self._Vspan_threshold:
                break
        log.debug("terminated after %d episodes", curr_episode + 1)
        log.debug("finished running the RL algorithm")
        # Create the RLDialog instance.
        policy_array = Qvalues.argmax(1)
        policy_dict = {}
        non_terminals = iter_config_states(self._config_values, True)
        for i, state in enumerate(non_terminals):
            action = int(policy_array[i])
            if action in state:
                # Ensure that the policy doesn't suggest questions
                # that have been already answered.
                action = next(iter((a for a in range(env.num_actions)
                                    if a not in state)))
            state_key = frozenset(state.items())
            policy_dict[state_key] = action
        dialog = RLDialog(self._config_values, rules, policy_dict,
                          validate=self._validate)
        return dialog


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
        raise NotImplementedError

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

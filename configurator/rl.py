"""Configuration dialogs based on reinforcement learning.
"""

import logging
import pprint
from functools import reduce
from operator import mul

import numpy as np
import numpy.ma as ma
from pybrain.auxiliary import GradientDescent
from pybrain.rl.agents import LearningAgent
from pybrain.rl.environments.environment import Environment
from pybrain.rl.environments.episodic import EpisodicTask
from pybrain.rl.experiments import EpisodicExperiment
from pybrain.rl.learners import Q
from pybrain.rl.learners.valuebased import ActionValueTable
from pybrain.rl.learners.valuebased.interface import ActionValueInterface
from pybrain.rl.learners.valuebased.valuebased import ValueBasedLearner
from pybrain.structure.modules import Module
from pybrain.structure import (FeedForwardNetwork, FullConnection,
                               BiasUnit, LinearLayer, SigmoidLayer)


from .dialogs import Dialog, DialogBuilder
from .util import iter_config_states


__all__ = ["RLDialogBuilder"]


log = logging.getLogger(__name__)


class RLDialogBuilder(DialogBuilder):
    """Build a configuration dialog using reinforcement learning.

    Arguments:
        consistency: Type of consistency check used to filter the
            domain of the remaining questions during the simulation of
            the RL episodes. Possible values are: `'global'` and
            `'local'` (only implemented for binary constraints). This
            argument is ignored for rule-based dialogs.
        rl_table: Representation of the action-value table. Possible
            values are `'exact'` (explicit representation of all the
            configuration states) and `'approximate'` (approximate
            representation using a neural network).
        rl_backprop_step_size: Gradient descent step size used to
            update the parameters of the neural network approximating
            the action-value table (only relevant if `rl_table` is
            `'approximate'`).
        rl_learning_rate: Q-learning learning rate.
        rl_initial_epsilon: Initial epsilon value for the
            epsilon-greedy exploration. The value decays linearly with
            the number of simulated episodes.
        rl_num_episodes: Number of simulated episodes.

    See :class:`configurator.dialogs.DialogBuilder` for the remaining
    arguments.
    """

    def __init__(self, var_domains, sample, rules=None, constraints=None,
                 consistency="local",
                 rl_table="approximate",
                 rl_backprop_step_size=0.1,
                 rl_learning_rate=0.3,
                 rl_initial_epsilon=0.5,
                 rl_num_episodes=1000,
                 validate=False):
        super().__init__(var_domains, sample, rules, constraints, validate)
        if consistency in {"global", "local"}:
            self._consistency = consistency
        else:
            raise ValueError("Invalid consistency value")
        if rl_table in {"exact", "approximate"}:
            self._rl_table = rl_table
        else:
            raise ValueError("Invalid rl_table value")
        self._rl_backprop_step_size = rl_backprop_step_size
        self._rl_learning_rate = rl_learning_rate
        self._rl_initial_epsilon = rl_initial_epsilon
        self._rl_num_episodes = rl_num_episodes

    def build_dialog(self):
        """Construct a configuration dialog.

        Returns:
            An instance of a `configurator.dialogs.Dialog` subclass.
        """
        dialog = Dialog(self.var_domains, self.rules, self.constraints)
        env = DialogEnvironment(dialog, self._consistency, self._freq_table)
        task = DialogTask(env)
        if self._rl_table == "exact":
            table = ExactQTable(self.var_domains)
            learner = ExactQLearning(self._rl_learning_rate)
        elif self._rl_table == "approximate":
            table = ApproxQTable(self.var_domains)
            learner = ApproxQLearning(self._rl_learning_rate,
                                      self._rl_backprop_step_size)
        agent = DialogAgent(table, learner, self._rl_initial_epsilon)
        exp = EpisodicExperiment(task, agent)
        log.info("running the RL algorithm")
        complete_episodes = 0
        for curr_episode in range(self._rl_num_episodes):
            log.info("epsilon value is %g", agent.epsilon)
            exp.doEpisodes(number=1)
            if env.dialog.is_complete():
                complete_episodes += 1
                agent.learn(episodes=1)
            agent.reset()
            agent.epsilon = (self._rl_initial_epsilon *
                             (self._rl_num_episodes - curr_episode - 1) /
                             self._rl_num_episodes)
        log.info("simulated %d episodes", self._rl_num_episodes)
        log.info("learned from %d episodes", complete_episodes)
        log.info("finished running the RL algorithm")
        # Create the RLDialog instance.
        dialog = RLDialog(self.var_domains, table, self.rules,
                          self.constraints, validate=self._validate)
        return dialog


class RLDialog(Dialog):
    """Configuration dialog generated using reinforcement learning.

    Arguments:
        table: A `ExactQTable` or `ApproxQTable` instance.

    See `configurator.dialogs.Dialog` for information about the
    remaining arguments, attributes and methods.
    """

    def __init__(self, var_domains, table,
                 rules=None, constraints=None, validate=False):
        self._table = table
        super().__init__(var_domains, rules, constraints, validate)

    def get_next_question(self):
        state = self._table.transformInput(config=self.config)
        action = self._table.getMaxAction(state, self.config)
        return action


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
        if log.isEnabledFor(logging.DEBUG):
            log.debug("current configuration in the environment:\n%s",
                      pprint.pformat(self.dialog.config))
        return self.dialog.config

    def performAction(self, action):
        log.debug("performing action %d in the environment", action)
        var_index = int(action)
        var_values = self.dialog.get_possible_answers(var_index)
        # Simulate the user response.
        probs = np.empty_like(var_values, dtype=np.float)
        for i, var_value in enumerate(var_values):
            response = {var_index: var_value}
            probs[i] = self._freq_table.cond_prob(response, self.dialog.config)
        bins = np.cumsum(probs / probs.sum())
        sampled_bin = np.random.random((1, ))
        var_value = var_values[int(np.digitize(sampled_bin, bins))]
        log.debug("simulated user response %r", var_value)
        self.dialog.set_answer(var_index, var_value, self._consistency)
        if log.isEnabledFor(logging.DEBUG):
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
        table: A `ExactQTable` or `ApproxQTable` instance.
        learner: A `ExactQLearning` or `ApproxQLearning` instance.
        epsilon: Epsilon value for the epsilon-greedy exploration.
    """

    def __init__(self, table, learner, epsilon):
        super().__init__(table, learner)  # self.module = table
        self.epsilon = epsilon

    def newEpisode(self):
        log.debug("new episode in the agent")
        super().newEpisode()
        self.lastconfig = None
        self.laststate = None
        self.lastaction = None
        self.lastreward = None

    def integrateObservation(self, config):  # Step 1
        self.lastconfig = config
        self.laststate = self.module.transformInput(config=config)

    def getAction(self):  # Step 2
        log.debug("getting an action from the agent")
        # Epsilon-greedy exploration. It ensures that a valid action
        # at the current configuration state is always returned
        # (i.e. a question that hasn't been answered).
        action = self.module.getMaxAction(self.laststate, self.lastconfig)
        log.debug("the greedy action is %d", action)
        if np.random.uniform() < self.epsilon:
            valid_actions = [a for a in range(self.module.numActions)
                             if a not in self.lastconfig]
            action = np.random.choice(valid_actions)
            log.debug("performing random action %d instead", action)
        self.lastaction = action
        return self.lastaction

    def giveReward(self, reward):  # Step 3
        self.lastreward = reward
        if self.logging:
            self.history.addSample(self.laststate, self.lastaction,
                                   self.lastreward)


class ExactQTable(ActionValueTable):

    def __init__(self, var_domains):
        self.var_domains = var_domains
        var_card = list(map(len, self.var_domains))
        # One variable value is added for the unknown state.
        all_states = reduce(mul, map(lambda x: x + 1, var_card))
        terminal_states = reduce(mul, map(len, self.var_domains))
        num_states = (all_states - terminal_states) + 1
        num_actions = len(self.var_domains)
        log.info("Q has %d states and %d actions", num_states, num_actions)
        super().__init__(num_states, num_actions)
        self.initialize(0)
        self.Q = self.params.reshape(num_states, num_actions)

    def transformInput(self, config=None, state=None):
        # Find the position of the state among all the possible
        # configuration states. This isn't efficient, but the tabular
        # version doesn't work for many variables anyway.
        assert config is None or state is None
        config_key = (None if config is None else
                      hash(frozenset(config.items())))
        non_terminals = iter_config_states(self.var_domains, True)
        for cur_state, cur_config in enumerate(non_terminals):
            if state == cur_state:
                return cur_config
            elif config_key == hash(frozenset(cur_config.items())):
                return cur_state

    def getMaxAction(self, state, config=None):
        if config is None:
            config = self.transformInput(state=state)
        invalid_action = [a in config for a in range(self.numActions)]
        action = ma.array(self.Q[state, :], mask=invalid_action).argmax()
        return action


class ApproxQTable(Module, ActionValueInterface):

    def __init__(self, var_domains):
        self.var_domains = var_domains
        self.numActions = len(self.var_domains)
        super().__init__(sum(map(len, self.var_domains)), 1)
        self.network = self._buildNetwork()
        log.info("the neural network has %d parameters",
                 len(self.network.params))

    def _buildNetwork(self):
        n = FeedForwardNetwork()
        n.addInputModule(LinearLayer(self.indim, name="input"))
        n.addModule(SigmoidLayer(self.numActions, name="hidden"))
        n.addOutputModule(SigmoidLayer(self.numActions, name="output"))
        n.addModule(BiasUnit(name="bias"))
        n.addConnection(FullConnection(n["input"], n["hidden"]))
        n.addConnection(FullConnection(n["bias"], n["hidden"]))
        n.addConnection(FullConnection(n["hidden"], n["output"]))
        n.addConnection(FullConnection(n["bias"], n["output"]))
        n.sortModules()  # N(0,1) weight initialization
        return n

    def transformInput(self, config=None, state=None):
        # Variables are represented using the 1-of-C encoding with -1
        # for unset values and 1 for the set value. If a variable is
        # not set, all the values are 0.
        assert config is None or state is None
        if state is None:
            state = []
            for var_index, var_values in enumerate(self.var_domains):
                if var_index in config:
                    input_values = -1 * np.ones((len(var_values), ))
                    k = var_values.index(config[var_index])
                    input_values[k] = 1
                else:
                    input_values = np.zeros((len(var_values), ))
                state.append(input_values)
            return np.concatenate(state)
        elif config is None:
            config = {}
            i = 0
            for var_index, var_values in enumerate(self.var_domains):
                input_values = state[i:i + len(var_values)]
                k = np.flatnonzero(input_values == 1)
                assert k.size in {0, 1}
                if k.size == 1:
                    config[var_index] = var_values[k]
                i += len(var_values)
            return config

    def transformOutput(self, Q=None, output=None):
        assert Q is None or output is None
        Q_max = self.numActions - 1
        if Q is None:
            Q_values = output * Q_max
            return Q_values
        else:
            output_values = Q / Q_max
            return output_values

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[0] = self.getMaxAction(inbuf)

    def getActionValues(self, state):
        output_values = self.network.activate(state)
        Q_values = self.transformOutput(output=output_values)
        return Q_values

    def getValue(self, state, action):
        Q_values = self.getActionValues(state)
        return Q_values[action]

    def getMaxAction(self, state, config=None):
        if config is None:
            config = self.transformInput(state=state)
        invalid_action = [a in config for a in range(self.numActions)]
        Q_values = self.getActionValues(state)
        action = ma.array(Q_values, mask=invalid_action).argmax()
        return action


class ExactQLearning(Q):

    def __init__(self, rl_learning_rate):
        super().__init__(alpha=rl_learning_rate, gamma=1.0)

    def learn(self):
        # We need to process the reward for entering the terminal
        # state but Q.learn doesn't do it. Let Q.learn process the
        # complete episode first, and then process the last
        # observation. We assume Q is zero for the terminal state.
        super().learn()
        # This will only work if episodes are processed one by one,
        # so ensure there's only one sequence in the dataset.
        assert self.batchMode and self.dataset.getNumSequences() == 1
        seq = next(iter(self.dataset))
        for laststate, lastaction, lastreward in seq:
            pass  # skip all the way to the last
        laststate = int(laststate)
        lastaction = int(lastaction)
        qvalue = self.module.getValue(laststate, lastaction)
        new_qvalue = qvalue + self.alpha * (lastreward - qvalue)
        self.module.updateValue(laststate, lastaction, new_qvalue)


# Based on Nees Jan van Eck, Michiel van Wezel, Application of
# reinforcement learning to the game of Othello, Computers &
# Operations Research, Volume 35, Issue 6, June 2008, Pages 1999-2017.
class ApproxQLearning(ValueBasedLearner):

    def __init__(self, rl_learning_rate, rl_backprop_step_size):
        super().__init__()
        self._rl_learning_rate = rl_learning_rate
        self._rl_backprop_step_size = rl_backprop_step_size

    def learn(self):
        # Assume episodes are processed one by one.
        assert self.batchMode and self.dataset.getNumSequences() == 1
        avg_error = np.zeros((self.module.network.outdim, ))
        laststate = None
        experience = list(next(iter(self.dataset)))
        for state, action, reward in experience:
            if laststate is None:
                laststate = state
                lastaction = int(action)
                lastreward = reward
                continue
            # Calculate the error in the Q values, scale it back to
            # the network's output and update the average error.
            Q_output = self.module.getValue(laststate, lastaction)
            max_a = self.module.getMaxAction(state)
            max_Q = self.module.getValue(state, max_a)
            Q_target = ((1 - self._rl_learning_rate) * Q_output +
                        self._rl_learning_rate * (lastreward + max_Q))
            avg_error[lastaction] += \
                0.5 * (self.module.transformOutput(Q=Q_target) -
                       self.module.transformOutput(Q=Q_output)) ** 2
            # Update experience for the next iteration.
            laststate = state
            lastaction = int(action)
            lastreward = reward
        # Process the reward for entering the terminal state.
        # We assume Q is zero for the terminal state.
        Q_output = self.module.getValue(laststate, lastaction)
        Q_target = Q_output + self._rl_learning_rate * (lastreward - Q_output)
        avg_error[lastaction] += \
            0.5 * (self.module.transformOutput(Q=Q_target) -
                   self.module.transformOutput(Q=Q_output)) ** 2
        avg_error /= len(experience)
        self._backprop(avg_error)

    def _backprop(self, output_error):
        log.info("neural network error:\n%s", output_error)
        self.module.network.resetDerivatives()
        self.module.network.backActivate(output_error)
        gradient_descent = GradientDescent()
        gradient_descent.alpha = self._rl_backprop_step_size
        gradient_descent.init(self.module.network.params)
        new_params = gradient_descent(self.module.network.derivs)
        self.module.network.params[:] = new_params

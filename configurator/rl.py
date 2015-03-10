"""Configuration dialogs based on reinforcement learning.
"""

import logging
import pprint
from functools import reduce
from operator import mul

import numpy as np
import numpy.ma as ma
from scipy import stats
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
from pybrain.structure import (FeedForwardNetwork, TanhLayer,
                               BiasUnit, FullConnection)

from .dialogs import Dialog, DialogBuilder
from .util import iter_config_states


__all__ = ["RLDialogBuilder"]


log = logging.getLogger(__name__)


class RLDialogBuilder(DialogBuilder):
    """Build a configuration dialog using reinforcement learning.

    Arguments:
        rl_table: Representation of the action-value table. Possible
            values are `'exact'` (explicit representation of all the
            configuration states) and `'approximate'` (approximate
            representation using a neural network).
        rl_consistency: Type of consistency check used to filter the
            domain of the remaining questions during the simulation of
            the RL episodes. Possible values are: `'global'` and
            `'local'` (only implemented for binary constraints). This
            argument is ignored for rule-based dialogs.
        rl_learning_rate: Q-learning learning rate.
        rl_epsilon: Epsilon value for the epsilon-greedy exploration.
        rl_num_episodes: Number of simulated episodes.

    See :class:`configurator.dialogs.DialogBuilder` for the remaining
    arguments.
    """

    def __init__(self, var_domains, rules=None, constraints=None, sample=None,
                 rl_table="approximate",
                 rl_consistency="local",
                 rl_learning_rate=0.3,
                 rl_epsilon=0.1,
                 rl_num_episodes=1000,
                 validate=False):
        super().__init__(var_domains, rules, constraints, sample, validate)
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
        self._rl_num_episodes = rl_num_episodes

    def build_dialog(self):
        """Construct a configuration dialog.

        Returns:
            An instance of a `configurator.dialogs.Dialog` subclass.
        """
        dialog = Dialog(self.var_domains, self.rules, self.constraints)
        env = DialogEnvironment(dialog, self._rl_consistency, self._freq_table)
        task = DialogTask(env)
        if self._rl_table == "exact":
            learner = ExactQLearning(self._rl_learning_rate)
            table = ExactQTable(self.var_domains)
        elif self._rl_table == "approximate":
            learner = ApproxQLearning(self._rl_learning_rate)
            table = ApproxQTable(self.var_domains)
        agent = DialogAgent(table, learner, self._rl_epsilon)
        exp = EpisodicExperiment(task, agent)
        log.info("running the RL algorithm")
        complete_episodes = 0
        for curr_episode in range(self._rl_num_episodes):
            exp.doEpisodes(number=1)
            if env.dialog.is_complete():
                complete_episodes += 1
                agent.learn(episodes=1)
            agent.reset()
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
        state = self._table.getState(self.config)
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
            if prob > 0:
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
        self._epsilon = epsilon

    def newEpisode(self):
        log.debug("new episode in the agent")
        super().newEpisode()
        self.lastconfig = None
        self.laststate = None
        self.lastaction = None
        self.lastreward = None

    def integrateObservation(self, config):  # Step 1
        self.lastconfig = config
        self.laststate = self.module.getState(config)

    def getAction(self):  # Step 2
        log.debug("getting an action from the agent")
        # Epsilon-greedy exploration. It ensures that a valid action
        # at the current configuration state is always returned
        # (i.e. a question that hasn't been answered).
        action = self.module.getMaxAction(self.laststate, self.lastconfig)
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
        self.Q[num_states - 1, :] = 0

    def _transformInput(self, config=None, state=None):
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

    def getState(self, config):
        return self._transformInput(config=config)

    def getMaxAction(self, state, config=None):
        # Get the question that should be asked at the given state.
        if config is None:
            config = self._transformInput(state=state)
        invalid_actions = [a in config for a in range(self.numActions)]
        action = ma.array(self.Q[state, :], mask=invalid_actions).argmax()
        return action


# Based on pybrain.rl.learners.valuebased.ActionValueNetwork.
class ApproxQTable(Module, ActionValueInterface):

    def __init__(self, var_domains):
        self.var_domains = var_domains
        super().__init__(len(self.var_domains), 1)
        self.numActions = len(self.var_domains)
        self.network = self._buildNetwork()
        log.info("the neural network has %d parameters",
                 len(self.network.params))

    def _buildNetwork(self):
        n = FeedForwardNetwork()
        n.addInputModule(TanhLayer(self.numActions, name="input"))
        n.addModule(BiasUnit(name="input_bias"))
        n.addModule(TanhLayer(self.numActions, name="hidden"))
        n.addModule(BiasUnit(name="hidden_bias"))
        n.addOutputModule(TanhLayer(self.numActions, name="output"))
        n.addConnection(FullConnection(n["input"], n["hidden"]))
        n.addConnection(FullConnection(n["input_bias"], n["hidden"]))
        n.addConnection(FullConnection(n["hidden"], n["output"]))
        n.addConnection(FullConnection(n["hidden_bias"], n["output"]))
        n.sortModules()  # N(0,1) weight initialization
        return n

    def transformInput(self, config):
        input_values = -1 * np.ones((self.numActions, ))
        for var_index in config.keys():
            input_values[var_index] = 1
        return input_values

    def transformOutput(self, Q=None, output=None):
        assert Q is None or output is None
        Q_max = self.numActions - 1
        if Q is None:
            Q_values = Q_max * (output + 1) / 2
            return Q_values
        else:
            output_values = 2 * (Q / Q_max) - 1
            return output_values

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[0] = self.getMaxAction(inbuf)

    def getState(self, config):
        return self.transformInput(config)

    def getActionValues(self, state):
        output_values = self.network.activate(state)
        Q_values = self.transformOutput(output=output_values)
        return Q_values

    def getValue(self, state, action):
        Q_values = self.getActionValues(state)
        return Q_values[action]

    def getMaxAction(self, state, config=None):
        Q_values = self.getActionValues(state)
        invalid_actions = [state[a] == 1 for a in range(self.numActions)]
        action = ma.array(Q_values, mask=invalid_actions).argmax()
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

    def __init__(self, rl_learning_rate):
        super().__init__()
        self.alpha = rl_learning_rate

    def _backpropagate(self, Q_target, Q_output, action):
        Q_error = np.zeros((self.module.network.outdim, ))
        Q_error[action] = Q_target - Q_output
        log.debug("backpropagating a Q error of %g for action %d",
                  Q_error[action], action)
        # Transform the Q error values to error values of the
        # neural network and backpropagate the error.
        output_error = self.module.transformOutput(Q=Q_error)
        self.module.network.resetDerivatives()
        self.module.network.backActivate(output_error)
        gradient_descent = GradientDescent()
        gradient_descent.alpha = 0.1
        gradient_descent.init(self.module.network.params)
        new_params = gradient_descent(self.module.network.derivs)
        self.module.network.params[:] = new_params

    def learn(self):
        # Assume episodes are processed one by one.
        assert self.batchMode and self.dataset.getNumSequences() == 1
        laststate = None
        seq = next(iter(self.dataset))
        for state, action, reward in seq:
            if laststate is None:
                laststate = state
                lastaction = int(action)
                lastreward = reward
                continue
            # Calculate the error in the Q values.
            Q_output = self.module.getValue(laststate, lastaction)
            max_Q = self.module.getValue(state, self.module.getMaxAction(state))
            Q_target = ((1 - self.alpha) * Q_output +
                        self.alpha * (lastreward + max_Q))
            self._backpropagate(Q_target, Q_output, lastaction)
            # Update experience for the next iteration.
            laststate = state
            lastaction = int(action)
            lastreward = reward
        # Process the reward for entering the terminal state.
        # We assume Q is zero for the terminal state.
        Q_output = self.module.getValue(laststate, lastaction)
        Q_target = Q_output + self.alpha * (lastreward - Q_output)
        self._backpropagate(Q_target, Q_output, lastaction)

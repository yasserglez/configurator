"""Configuration dialogs based on reinforcement learning.
"""

import logging
import pprint
from functools import reduce
from operator import mul

import numpy as np
import numpy.ma as ma
from fann2 import libfann
from pybrain.datasets import ReinforcementDataSet
from pybrain.rl.agents import LearningAgent
from pybrain.rl.environments.environment import Environment
from pybrain.rl.environments.episodic import EpisodicTask
from pybrain.rl.experiments import EpisodicExperiment
from pybrain.rl.learners import Q
from pybrain.rl.learners.valuebased import ActionValueTable
from pybrain.rl.learners.valuebased.interface import ActionValueInterface
from pybrain.rl.learners.valuebased.valuebased import ValueBasedLearner
from pybrain.structure.modules import Module

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
        num_episodes: Number of simulated episodes.
        learning_batch: Collect experience from this many episodes
            before updating the action-value table.
        table: Representation of the action-value table. Possible
            values are `'exact'` (explicit representation of all the
            configuration states) and `'approx'` (approximate
            representation using a neural network).
        epsilon: Epsilon value for the epsilon-greedy exploration.
        learning_rate: Q-learning learning rate. This argument is used
            only with the exact action-value table. The approximate
            representation is learned using Neural Fitted Q-iteration.
        rprop_epochs: Maximum number of epochs of Rprop training. This
            argument is used only with the approximate action-value
            table representation.
        rprop_error: Rprop error threshold. This argument is used only
            with the approximate action-value table representation.

    See :class:`configurator.dialogs.DialogBuilder` for the remaining
    arguments.
    """

    def __init__(self, var_domains, sample, rules=None, constraints=None,
                 consistency="local",
                 num_episodes=1000,
                 learning_batch=1,
                 table="approx",
                 epsilon=0.1,
                 learning_rate=0.3,
                 rprop_epochs=100,
                 rprop_error=0.01,
                 validate=False):
        super().__init__(var_domains, sample, rules, constraints, validate)
        if consistency not in {"global", "local"}:
            raise ValueError("Invalid consistency value")
        if table not in {"exact", "approx"}:
            raise ValueError("Invalid table value")
        self._consistency = consistency
        self._num_episodes = num_episodes
        self._learning_batch = learning_batch
        self._table = table
        self._epsilon = epsilon
        self._learning_rate = learning_rate
        self._rprop_epochs = rprop_epochs
        self._rprop_error = rprop_error

    def build_dialog(self):
        """Construct a configuration dialog.

        Returns:
            An instance of a `configurator.dialogs.Dialog` subclass.
        """
        dialog = Dialog(self.var_domains, self.rules, self.constraints)
        env = DialogEnvironment(dialog, self._consistency, self._freq_table)
        task = DialogTask(env)
        if self._table == "exact":
            table = ExactQTable(self.var_domains)
            learner = ExactQLearning(self._learning_rate)
        elif self._table == "approx":
            table = ApproxQTable(self.var_domains)
            learner = ApproxQLearning(self._rprop_epochs,
                                      self._rprop_error)
        agent = DialogAgent(table, learner, self._epsilon)
        exp = DialogExperiment(task, agent)
        log.info("running the RL algorithm")
        simulated_episodes = 0
        complete_episodes = 0
        while simulated_episodes < self._num_episodes:
            exp.doEpisodes(number=self._learning_batch)
            simulated_episodes += self._learning_batch
            complete_episodes += agent.history.getNumSequences()
            agent.learn()
            agent.reset()  # clear the previous batch
        log.info("simulated %d episodes", simulated_episodes)
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
        state = self._table.transformState(config=self.config)
        action = self._table.getMaxAction(state, self.config)
        return action


class DialogExperiment(EpisodicExperiment):

    def doEpisodes(self, number):
        # Overriding to handle skipping inconsistent episodes.
        log.info("simulating %d episodes", number)
        cumrewards = []
        for episode in range(number):
            self.agent.newEpisode()
            self.task.reset()
            while not self.task.isFinished():
                self._oneInteraction()
            if self.task.env.dialog.is_consistent():
                cumrewards.append(self.task.cumreward)
            else:
                index = self.agent.history.getNumSequences() - 1
                self.agent.history.removeSequence(index)
        log.info("finished %d complete episodes, average total reward %g",
                 len(cumrewards), np.mean(cumrewards))
        return cumrewards


class DialogEnvironment(Environment):

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
        if not self.env.dialog.is_consistent():
            log.debug("finished an episode, reached an inconsistent state")
            is_finished = True
        elif self.env.dialog.is_complete():
            log.debug("finished an episode, total reward %d", self.cumreward)
            is_finished = True
        return is_finished


class DialogAgent(LearningAgent):

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
        self.laststate = self.module.transformState(config=config)

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

    def transformState(self, config=None, state=None):
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
            config = self.transformState(state=state)
        invalid_action = [a in config for a in range(self.numActions)]
        action = ma.array(self.Q[state, :], mask=invalid_action).argmax()
        return action


class ApproxQTable(Module, ActionValueInterface):

    def __init__(self, var_domains):
        self.var_domains = var_domains
        super().__init__(sum(map(len, self.var_domains)), 1)
        self.numActions = len(self.var_domains)
        self.network = self._buildNetwork()

    def _buildNetwork(self):
        num_input = self.indim + len(self.var_domains)
        num_hidden = len(self.var_domains)
        num_output = self.outdim
        net = libfann.neural_net()
        net.create_standard_array((num_input, num_hidden, num_output))
        net.set_training_algorithm(libfann.TRAIN_RPROP)
        net.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC)
        net.set_activation_function_output(libfann.SIGMOID_SYMMETRIC)
        net.set_training_algorithm(libfann.TRAIN_RPROP)
        net.set_train_error_function(libfann.ERRORFUNC_LINEAR)
        net.set_train_stop_function(libfann.STOPFUNC_MSE)
        # TODO: Couldn't get set_weight_array to work.
        total_neurons = net.get_total_neurons()
        for i in range(total_neurons):
            for j in range(i + 1, total_neurons):
                weight = np.random.uniform(-0.5, 0.5)
                net.set_weight(i, j, weight)
        log.info("the neural network has %d weights",
                 net.get_total_connections())
        return net

    def transformState(self, config=None, state=None):
        # Each variable is represented using dummy variables
        # (1-of-C encoding) with 0 as -1 and 1 as 1.
        assert config is None or state is None
        if state is None:
            state = []
            for var_index, var_values in enumerate(self.var_domains):
                input_values = -1 * np.ones((len(var_values), ))
                if var_index in config:
                    var_value = config[var_index]
                    k = var_values.index(var_value)
                    input_values[k] = 1
                state.append(input_values)
            return np.concatenate(state)
        elif config is None:
            i = 0
            config = {}
            for var_index, var_values in enumerate(self.var_domains):
                input_values = state[i:i + len(var_values)]
                k = np.flatnonzero(input_values == 1)
                assert k.size in {0, 1}
                if k.size == 1:
                    config[var_index] = var_values[k]
                i += len(var_values)
            return config

    def transformInput(self, state, action):
        # The action is represented using dummy variables
        # (1-of-C encoding) with 0 as -1 and 1 as 1.
        action_values = -1 * np.ones((len(self.var_domains), ))
        action_values[action] = 1
        input_values = np.concatenate([state, action_values])
        return input_values

    def transformOutput(self, Q=None, output=None):
        # Scale neural net output from [-1, 1] to [0, Q_max].
        assert Q is None or output is None
        Q_max = len(self.var_domains) - 1  # at least one question asked
        if Q is None:
            Q_values = Q_max * (output + 1) / 2
            return Q_values
        else:
            output_values = 2 * (Q / Q_max) - 1
            return output_values

    def getValue(self, state, action):
        input_values = self.transformInput(state, action)
        output_value = self.network.run(input_values)[0]
        Q_value = self.transformOutput(output=output_value)
        return Q_value

    def _getMaxActionValue(self, state, config=None):
        if config is None:
            config = self.transformState(state=state)
        max_action, max_value = None, None
        for action in range(len(self.var_domains)):
            if action not in config:
                value = self.getValue(state, action)
                if max_action is None or value > max_value:
                    max_action = action
                    max_value = value
        return max_action, max_value

    def getMaxAction(self, state, config=None):
        return self._getMaxActionValue(state, config)[0]

    def getMaxValue(self, state, config=None):
        return self._getMaxActionValue(state, config)[1]


class ExactQLearning(Q):

    def __init__(self, learning_rate):
        super().__init__(alpha=learning_rate, gamma=1.0)

    def learn(self):
        # Q.learn doesn't process the reward for entering the terminal
        # state. Let Q.learn process the complete episode first, and
        # then process the last observation. This will work only if
        # episodes are processed one by one, so ensure there's only
        # one sequence in the dataset.
        assert self.batchMode and self.dataset.getNumSequences() == 1
        super().learn()
        seq = next(iter(self.dataset))
        for laststate, lastaction, lastreward in seq:
            pass  # skip all the way to the end
        laststate = int(laststate)
        lastaction = int(lastaction)
        # Assuming Q is zero for the terminal state.
        qvalue = self.module.getValue(laststate, lastaction)
        new_qvalue = qvalue + self.alpha * (lastreward - qvalue)
        self.module.updateValue(laststate, lastaction, new_qvalue)


# Neural Fitted Q-iteration.
class ApproxQLearning(ValueBasedLearner):

    def __init__(self, rprop_epochs, rprop_error):
        super().__init__()
        self._rprop_epochs = rprop_epochs
        self._rprop_error = rprop_error
        self._samples = None

    def learn(self):
        # Extend the incremental dataset with the new transitions.
        if self._samples is None:
            # The agent sets the module (i.e. the ApproxQTable
            # instance) after the object is created.
            self._samples = ReinforcementDataSet(self.module.indim,
                                                 self.module.outdim)
        for episode in self.dataset:
            self._samples.newSequence()
            for state, action, reward in episode:
                self._samples.addSample(state, action, reward)
        # Train the neural network.
        input_values, target_values = self._generate_pattern_set()
        self._rprop_training(input_values, target_values)

    def _generate_pattern_set(self):
        input_values, target_values = [], []
        for episode in self._samples:
            laststate = None
            for state, action, reward in episode:
                if laststate is None:
                    laststate = state
                    lastaction = int(action)
                    lastreward = reward
                    continue
                # Build Q(s,a) = r(s,a) + max Q(s',a') from
                # (laststate, lastaction, lastreward, state).
                input_value = self.module.transformInput(laststate, lastaction)
                Q_value = lastreward + self.module.getMaxValue(state)
                target_value = self.module.transformOutput(Q=Q_value)
                input_values.append(input_value)
                target_values.append([target_value])
                # Prepare for the next iteration.
                laststate = state
                lastaction = int(action)
                lastreward = reward
            # Add the reward for entering the terminal state.
            # Assuming Q is zero for the terminal state.
            input_value = self.module.transformInput(laststate, lastaction)
            target_value = self.module.transformOutput(Q=lastreward)
            input_values.append(input_value)
            target_values.append([target_value])
        return input_values, target_values

    def _rprop_training(self, input_values, target_values):
        log.info("training the neural network using Rprop")
        net = self.module.network
        data = libfann.training_data()
        data.set_train_data(input_values, target_values)
        log.info("the training set contains %d samples",
                 data.length_train_data())
        net.reset_MSE()
        net.train_on_data(data, self._rprop_epochs, 0, self._rprop_error)
        log.info("final training MSE is %g", net.get_MSE())

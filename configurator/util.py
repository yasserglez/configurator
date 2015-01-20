"""
Utility Functions
"""

import math
import time
from functools import reduce
from operator import mul

import numpy as np
import pandas as pd
from sortedcontainers import SortedSet
from sklearn.cross_validation import KFold
from sklearn.utils import check_random_state


def load_config_sample(csv_file, dtype=np.uint8):
    """Load a CSV file with a sample of categorical variables.

    Read the CSV file and return an equivalent numpy array with the
    different categorical values represented by integers.

    Arguments:
        csv_file: Path to a CSV file.
        dtype: dtype of the returned numpy array
            (default: numpy.uint8).

    Returns:
        A 2-dimensional numpy array.
    """
    df = pd.read_csv(csv_file)
    config_sample = np.zeros(df.shape, dtype=dtype)
    for j, column in enumerate(df):
        df[column] = df[column].astype("category")
        config_sample[:, j] = df[column].cat.codes
    return config_sample


def simulate_configurator(configurator, config):
    """Simulate a configurator.

    Simulate the use of the configurator to predict the given
    configuration.

    Arguments:
        configurator: An instance of a Configurator subclass.
        config: A complete configuration, i.e. a dict mapping variable
            indices to their values.

    Returns:
        A tuple with two elements. The first elements gives the
        accuracy of the prediction, the second the number of questions
        that were asked (both normalized in [0,1]).
    """
    accuracy, questions = 0, 0
    configurator.reset()
    while not configurator.is_complete():
        var_index = configurator.get_next_question()
        configurator.set_answer(var_index, config[var_index])
        questions += 1
    for var_index in config.keys():
        if configurator.config[var_index] == config[var_index]:
            accuracy += 1
    # Normalize the measures and return.
    questions /= len(config)
    accuracy /= len(config)
    return accuracy, questions


def cross_validation(n_folds, random_state, builder_class, builder_kwargs,
                     config_sample, config_values=None):
    """Measure the performance of a configurator builder.

    Use the configurator builder to perform a k-folds cross validation on
    the given configuration sample. The sample is shuffled before
    dividing it into batches. The performance is measured in terms of
    the accuracy of the predicted configuration and the number of
    questions that were asked.

    Arguments:
        n_folds: Number of folds. Must be at least 2.
        random_state: Pseudo-random number generator state (int or
            numpy.random.RandomState) used for random sampling. If
            None, use default numpy RNG for shuffling.
        builder_class: A ConfiguratorBuilder subclass.
        builder_kwargs: A dict with arguments to pass to builder_class
            when a new instance is created (except config_sample and
            config_values).
        config_sample: A 2-dimensional numpy array containing a sample
            of the configuration variables.
        config_values: A list with one entry for each variable,
            containing an enumerable with all the possible values of
            the variable. If it is not given, it is automatically
            computed from the columns of config_sample.

    Returns:
        A pandas.DataFrame with one row for each fold and one column
        for each one of the following statistics: mean and standard
        deviation of the prediction accuracy, mean and standard
        deviation of the number of questions that were asked
        (normalized by the total number of questions).
    """
    if config_values is None:
        config_values = [list(SortedSet(config_sample[:, i]))
                         for i in range(config_sample.shape[1])]
    # Copy builder_kwargs to avoid inserting config_sample and
    # config_values into the original dict.
    builder_kwargs = builder_kwargs.copy()
    builder_kwargs["config_values"] = config_values
    # Initialize the output df and collect the statistics.
    result = pd.DataFrame(np.zeros((n_folds, 4)),
                          columns=["accuracy_mean", "accuracy_std",
                                   "questions_mean", "questions_std"])
    k = 0  # current fold index
    folds = KFold(config_sample.shape[0], n_folds=n_folds,
                  shuffle=True, random_state=random_state)
    for train_indices, test_indices in folds:
        # Build the configurator using the training sample.
        train_sample = config_sample[train_indices, :]
        builder_kwargs["config_sample"] = train_sample
        builder = builder_class(**builder_kwargs)
        configurator = builder.build_configurator()
        # Collect the results from the testing sample.
        test_sample = config_sample[test_indices, :]
        accuracy_results, questions_results = [], []
        for i in range(test_sample.shape[0]):
            config = {j: test_sample[i, j] for j in range(len(config_values))}
            accuracy, questions = simulate_configurator(configurator, config)
            accuracy_results.append(accuracy)
            questions_results.append(questions)
        # Summarize the results.
        result.loc[k, "accuracy_mean"] = np.mean(accuracy_results)
        result.loc[k, "accuracy_std"] = np.std(accuracy_results)
        result.loc[k, "questions_mean"] = np.mean(questions_results)
        result.loc[k, "questions_std"] = np.std(questions_results)
        k += 1  # move to the next fold
    return result


def measure_scalability(random_state, builder_class, builder_kwargs,
                        config_sample, config_values=None):
    """Measure the scalability of a configurator builder.

    Use the configurator builder to construct a sequence of
    configurators with an increasing number of variables from the
    configuration sample, measuring the CPU time on each case. The
    variables in the configuration sample are added in a random order.

    Arguments:
        random_state: Pseudo-random number generator state (int or
            numpy.random.RandomState) used for random sampling. If
            None, use default numpy RNG for shuffling.
        builder_class: A ConfiguratorBuilder subclass.
        builder_kwargs: A dict with arguments to pass to builder_class
            when a new instance is created (except config_sample and
            config_values).
        config_sample: A 2-dimensional numpy array containing a sample
            of the configuration variables.
        config_values: A list with one entry for each variable,
            containing an enumerable with all the possible values of
            the variable. If it is not given, it is automatically
            computed from the columns of config_sample.

    Returns:
        A pandas.DataFrame with two columns. The first column gives
        the number of binary variables and the second the
        corresponding measured CPU time (in seconds). The number of
        binary variables is computed as the log2 of the number of
        possible configurations.
    """
    if config_values is None:
        config_values = [list(SortedSet(config_sample[:, i]))
                         for i in range(config_sample.shape[1])]
    # Copy builder_kwargs to avoid inserting config_sample and
    # config_values into the original dict.
    builder_kwargs = builder_kwargs.copy()
    # Choose the order of the variables.
    num_vars = len(config_values)
    var_order = np.arange(num_vars)
    rng = check_random_state(random_state)
    rng.shuffle(var_order)
    # Initialize the output df.
    result = pd.DataFrame({"bin_vars": np.zeros(num_vars - 1),
                           "cpu_time": np.zeros(num_vars - 1)})
    for i in range(2, num_vars + 1):
        if i == num_vars:
            curr_config_sample = config_sample
            curr_config_values = config_values
        else:
            curr_vars = var_order[:i]
            curr_config_sample = config_sample[:, curr_vars]
            curr_config_values = [config_values[v] for v in curr_vars]
        curr_config_card = reduce(mul, map(len, curr_config_values))
        builder_kwargs["config_sample"] = curr_config_sample
        builder_kwargs["config_values"] = curr_config_values
        t_start = time.process_time()  # start
        builder = builder_class(**builder_kwargs)
        builder.build_configurator()
        t_end = time.process_time()  # stop
        result.loc[i - 2, "bin_vars"] = math.log2(curr_config_card)
        result.loc[i - 2, "cpu_time"] = t_end - t_start
    return result

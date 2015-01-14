"""
Utility Functions
"""

from sortedcontainers import SortedSet
from sklearn.cross_validation import KFold


def simulate_dialog(dialog, config):
    """Simulate the use of the dialog to predict the given configuration.

    Arguments:
        dialog: An instance of a ConfigDialog subclass.
        config: A complete configuration, i.e. a dict mapping variable
            indices to their values.

    Returns:
        A tuple with two elements. The first elements gives the
        accuracy of the prediction, the second the number of questions
        that were asked (both normalized in [0,1]).
    """
    accuracy, questions = 0, 0
    dialog.reset()
    while not dialog.is_complete():
        var_index = dialog.get_next_question()
        dialog.set_answer(var_index, config[var_index])
        questions += 1
    for var_index in config.keys():
        if dialog.config[var_index] == config[var_index]:
            accuracy += 1
    # Normalize the measures and return.
    questions /= len(config)
    accuracy /= len(config)
    return accuracy, questions



def cross_validation(n_folds, random_state, buildier_class, builder_kwargs,
                     config_sample, config_values=None):
    """Measure the performance of a dialog builder.

    Use the dialog builder to perform a k-folds cross validation on
    the given configuration sample. The sample is shuffled before
    dividing it into batches. The performance is measured in terms of
    the accuracy of the predicted configuration and the number of
    questions that were asked.

    Arguments:
        n_folds: Number of folds. Must be at least 2.
        random_state: Pseudo-random number generator state (int or
            numpy.random.RandomState) used for random sampling. If
            None, use default numpy RNG for shuffling.
        builder_class: A ConfigDialogBuilder subclass.
        builder_kwargs: A dictionary with arguments to pass to
            buildier_class when a new instance is created (except
            config_sample and config_values).
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


def measure_scalability(random_state, builder_class, builder_kwargs,
                        config_sample, config_values):
    """Measure the scalability of a dialog builder.

    Use the dialog builder to construct a sequence of dialogs with an
    increasing number of variables from the configuration sample,
    measuring the CPU time on each case. The variables in the
    configuration sample are added in a random order.

    Arguments:
        random_state: Pseudo-random number generator state (int or
            numpy.random.RandomState) used for random sampling. If
            None, use default numpy RNG for shuffling.
        builder_class: A ConfigDialogBuilder subclass.
        builder_kwargs: A dictionary with arguments to pass to
            buildier_class when a new instance is created (except
            config_sample and config_values).
        config_sample: A 2-dimensional numpy array containing a sample
            of the configuration variables.
        config_values: A list with one entry for each variable,
            containing an enumerable with all the possible values of
            the variable. If it is not given, it is automatically
            computed from the columns of config_sample.

    Returns:
        A pandas.DataFrame with two columns. The first column gives
        the number of binary variables and the second the
        corresponding measured CPU time. The number of binary
        variables is computed as the log2 of the number of possible
        configurations.
    """

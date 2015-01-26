import os
import sys
import logging
import random

import numpy as np

from .examples import load_email_client, load_titanic
from ..policy import DPConfigDialogBuilder, RLConfigDialogBuilder
from ..util import (load_config_sample, simulate_dialog,
                    cross_validation, measure_scalability)


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.handlers[0].setFormatter(logging.Formatter("%(asctime)s:%(message)s"))


def test_load_config_sample():
    original_sample = load_titanic()
    tests_dir = os.path.abspath(os.path.dirname(__file__))
    csv_file = os.path.join(tests_dir, "titanic.csv")
    loaded_sample = load_config_sample(csv_file)
    assert np.issubdtype(loaded_sample.dtype, np.integer)
    assert loaded_sample.shape == original_sample.shape
    for j in range(loaded_sample.shape[1]):
        original_j_labels = np.unique(original_sample[:, j])
        loaded_j_labels = np.unique(loaded_sample[:, j])
        assert len(loaded_j_labels) == len(original_j_labels)


def _test_simulate_dialog(builder_class, builder_kwargs):
    print("", file=sys.stderr)  # newline before the logging output
    random_state = 42; random.seed(random_state); np.random.seed(random_state)
    email_client = load_email_client(as_integers=True)
    builder = builder_class(
        config_sample=email_client.config_sample,
        assoc_rule_min_support=email_client.min_support,
        assoc_rule_min_confidence=email_client.min_confidence,
        **builder_kwargs)
    dialog = builder.build_dialog()
    # {yes = 1, no = 0}, {smi = 1, lgi = 0}
    config = {1: 0, 0: 0}
    accuracy, questions = simulate_dialog(dialog, config)
    assert (accuracy, questions) == (1.0, 0.5)
    config = {1: 0, 0: 1}
    accuracy, questions = simulate_dialog(dialog, config)
    assert (accuracy, questions) == (0.5, 0.5)
    config = {1: 1, 0: 0}
    accuracy, questions = simulate_dialog(dialog, config)
    assert (accuracy, questions) == (1.0, 1.0)
    config = {1: 1, 0: 1}
    accuracy, questions = simulate_dialog(dialog, config)
    assert (accuracy, questions) == (1.0, 1.0)


def test_simulate_dp_policy_dialog():
    _test_simulate_dialog(DPConfigDialogBuilder, {})


def test_simulate_rl_policy_dialog():
    builder_kwargs = {"rl_episodes": 10}
    _test_simulate_dialog(RLConfigDialogBuilder, builder_kwargs)


def test_cross_validation():
    print("", file=sys.stderr)  # newline before the logging output
    email_client = load_email_client()
    n_folds = 10
    random_state = 42
    builder_class = DPConfigDialogBuilder
    builder_kwargs = {"assoc_rule_min_support": email_client.min_support,
                      "assoc_rule_min_confidence": email_client.min_confidence}
    df = cross_validation(n_folds, random_state, builder_class,
                          builder_kwargs, email_client.config_sample)
    assert len(df.index) == n_folds
    assert ((0.5 <= df["accuracy_mean"]) & (df["accuracy_mean"] <= 1)).all()
    assert ((0 <= df["accuracy_std"]) & (df["accuracy_std"] <= 0.25)).all()
    assert ((0.5 <= df["questions_mean"]) & (df["questions_mean"] <= 1)).all()
    assert ((0 <= df["questions_std"]) & (df["questions_std"] <= 0.25)).all()


def _test_measure_scalability(builder_class, builder_kwargs):
    print("", file=sys.stderr)  # newline before the logging output
    random_state = 42; random.seed(random_state); np.random.seed(random_state)
    config_sample = load_titanic()
    builder_kwargs.update({"assoc_rule_min_support": 0.5,
                           "assoc_rule_min_confidence": 0.9})
    df = measure_scalability(random_state, builder_class,
                             builder_kwargs, config_sample)
    assert df.shape == (config_sample.shape[1] - 1, 2)
    assert (df["bin_vars"] > 0).all()
    assert (df["cpu_time"] > 0).all()


def _test_scalability_mdp(algorithm, discard_states,
                          partial_assoc_rules, collapse_terminals):
    builder_class = DPConfigDialogBuilder
    builder_kwargs = {"dp_algorithm": algorithm,
                      "dp_discard_states": discard_states,
                      "dp_partial_assoc_rules": partial_assoc_rules,
                      "dp_collapse_terminals": collapse_terminals,
                      "dp_validate": True}
    _test_measure_scalability(builder_class, builder_kwargs)


def test_scalability_value_iteration_without_optim():
    _test_scalability_mdp("value-iteration", False, False, False)


def test_scalability_policy_iteration_without_optim():
    _test_scalability_mdp("policy-iteration", False, False, False)


def test_scalability_value_iteration_with_optim():
    _test_scalability_mdp("value-iteration", True, True, True)


def test_scalability_policy_iteration_with_optim():
    _test_scalability_mdp("policy-iteration", True, True, True)

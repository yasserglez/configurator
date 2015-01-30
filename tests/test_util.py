import os

import numpy as np

from configurator.dp import DPDialogBuilder
from configurator.rl import RLDialogBuilder
from configurator.util import (load_config_sample, simulate_dialog,
                               cross_validation, measure_scalability)


def test_load_config_sample(titanic_data):
    tests_dir = os.path.abspath(os.path.dirname(__file__))
    csv_file = os.path.join(tests_dir, "titanic.csv")
    loaded_sample = load_config_sample(csv_file)
    assert np.issubdtype(loaded_sample.dtype, np.integer)
    assert loaded_sample.shape == titanic_data.shape
    for j in range(loaded_sample.shape[1]):
        original_j_labels = np.unique(titanic_data[:, j])
        loaded_j_labels = np.unique(loaded_sample[:, j])
        assert len(loaded_j_labels) == len(original_j_labels)


def _test_simulate_dialog(builder_class, builder_kwargs, email_client):
    builder = builder_class(
        config_sample=email_client.config_sample,
        validate=True,
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


def test_simulate_dp_dialog(email_client):
    _test_simulate_dialog(DPDialogBuilder, {}, email_client)


def test_simulate_exact_rl_dialog(email_client):
    builder_kwargs = {"rl_table": "exact"}
    _test_simulate_dialog(RLDialogBuilder, builder_kwargs, email_client)


def test_simulate_approx_rl_dialog(email_client):
    builder_kwargs = {"rl_table": "approx"}
    _test_simulate_dialog(RLDialogBuilder, builder_kwargs, email_client)


def test_cross_validation(email_client):
    n_folds = 10
    random_state = 42
    builder_class = DPDialogBuilder
    builder_kwargs = {"validate": True,
                      "assoc_rule_min_support": email_client.min_support,
                      "assoc_rule_min_confidence": email_client.min_confidence}
    df = cross_validation(n_folds, random_state, builder_class,
                          builder_kwargs, email_client.config_sample)
    assert len(df.index) == n_folds
    assert ((0.5 <= df["accuracy_mean"]) & (df["accuracy_mean"] <= 1)).all()
    assert ((0 <= df["accuracy_std"]) & (df["accuracy_std"] <= 0.25)).all()
    assert ((0.5 <= df["questions_mean"]) & (df["questions_mean"] <= 1)).all()
    assert ((0 <= df["questions_std"]) & (df["questions_std"] <= 0.25)).all()


def _test_measure_scalability(builder_class, builder_kwargs,
                              titanic_sample, random_seed):
    builder_kwargs.update({"validate": True,
                           "assoc_rule_min_support": 0.5,
                           "assoc_rule_min_confidence": 0.9})
    df = measure_scalability(random_seed, builder_class,
                             builder_kwargs, titanic_sample)
    assert df.shape == (titanic_sample.shape[1] - 1, 2)
    assert (df["bin_vars"] > 0).all()
    assert (df["cpu_time"] > 0).all()


def _test_scalability_dp(algorithm, discard_states,
                         partial_assoc_rules, aggregate_terminals,
                         titanic_sample, random_seed):
    builder_class = DPDialogBuilder
    builder_kwargs = {"dp_algorithm": algorithm,
                      "dp_discard_states": discard_states,
                      "dp_partial_assoc_rules": partial_assoc_rules,
                      "dp_aggregate_terminals": aggregate_terminals}
    _test_measure_scalability(builder_class, builder_kwargs,
                              titanic_sample, random_seed)


def test_scalability_value_iteration_without_optim(titanic_sample, random_seed):
    _test_scalability_dp("value-iteration", False, False, False,
                         titanic_sample, random_seed)


def test_scalability_policy_iteration_without_optim(titanic_sample, random_seed):
    _test_scalability_dp("policy-iteration", False, False, False,
                         titanic_sample, random_seed)


def test_scalability_value_iteration_with_optim(titanic_sample, random_seed):
    _test_scalability_dp("value-iteration", True, True, True,
                         titanic_sample, random_seed)


def test_scalability_policy_iteration_with_optim(titanic_sample, random_seed):
    _test_scalability_dp("policy-iteration", True, True, True,
                         titanic_sample, random_seed)


def _test_scalability_rl(algorithm, table, titanic_sample, random_seed):
    builder_class = RLDialogBuilder
    builder_kwargs = {"rl_algorithm": algorithm,
                      "rl_table": table}
    _test_measure_scalability(builder_class, builder_kwargs,
                              titanic_sample, random_seed)


def test_scalability_qlearning_exact(titanic_sample, random_seed):
    _test_scalability_rl("q-learning", "exact",
                         titanic_sample, random_seed)


def test_scalability_qlearning_approx(titanic_sample, random_seed):
    _test_scalability_rl("q-learning", "approx",
                         titanic_sample, random_seed)


def test_scalability_sarsa_exact(titanic_sample, random_seed):
    _test_scalability_rl("sarsa", "exact",
                         titanic_sample, random_seed)


def test_scalability_sarsa_approx(titanic_sample, random_seed):
    _test_scalability_rl("sarsa", "approx",
                         titanic_sample, random_seed)

import os

import numpy as np

from .examples import load_email_client, load_titanic
from ..policy import MDPDialogBuilder
from ..util import (load_config_sample, simulate_dialog,
                    cross_validation, measure_scalability)


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


def test_simulate_dialog():
    email_client = load_email_client()
    builder = MDPDialogBuilder(
        config_sample=email_client.config_sample,
        assoc_rule_min_support=email_client.min_support,
        assoc_rule_min_confidence=email_client.min_confidence)
    dialog = builder.build_dialog()
    accuracy, questions = simulate_dialog(dialog, email_client.config)
    assert accuracy == 1.0
    assert questions == 0.5


def test_cross_validation():
    email_client = load_email_client()
    n_folds = 10
    random_state = 42
    builder_class = MDPDialogBuilder
    builder_kwargs = {"assoc_rule_min_support": email_client.min_support,
                      "assoc_rule_min_confidence": email_client.min_confidence}
    df = cross_validation(n_folds, random_state, builder_class,
                          builder_kwargs, email_client.config_sample)
    assert len(df.index) == n_folds
    assert ((0.5 <= df["accuracy_mean"]) & (df["accuracy_mean"] <= 1)).all()
    assert ((0 <= df["accuracy_std"]) & (df["accuracy_std"] <= 0.25)).all()
    assert (df["questions_mean"] == 0.5).all()
    assert (df["questions_std"] == 0).all()


def _test_measure_scalability(builder_class, builder_kwargs):
    random_state = 42
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
    builder_class = MDPDialogBuilder
    builder_kwargs = {"mdp_algorithm": algorithm,
                      "mdp_discard_states": discard_states,
                      "mdp_partial_assoc_rules": partial_assoc_rules,
                      "mdp_collapse_terminals": collapse_terminals,
                      "mdp_validate": True}
    _test_measure_scalability(builder_class, builder_kwargs)


def test_scalability_value_iteration_without_optim():
    _test_scalability_mdp("value-iteration", False, False, False)


def test_scalability_policy_iteration_without_optim():
    _test_scalability_mdp("policy-iteration", False, False, False)


def test_scalability_value_iteration_with_optim():
    _test_scalability_mdp("value-iteration", True, True, True)


def test_scalability_policy_iteration_with_optim():
    _test_scalability_mdp("policy-iteration", True, True, True)

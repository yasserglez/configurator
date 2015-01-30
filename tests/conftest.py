import os
import sys
import random
import logging
from collections import namedtuple

import pytest
import numpy as np

from configurator.util import load_config_sample


TESTS_DIR = os.path.abspath(os.path.dirname(__file__))


@pytest.fixture(scope="function", autouse=True)
def random_seed():
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    return random_seed


@pytest.fixture(scope="function", autouse=True)
def logger():
    logger = logging.getLogger("configurator")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s:%(message)s")
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    print("", file=sys.stderr)  # newline before the logging output
    return logger


@pytest.fixture(scope="session")
def titanic_data():
    # R's Titanic dataset. Contingency table expanded into explicit
    # form using the expand.table function from the epitools package.
    fname = os.path.join(TESTS_DIR, "titanic.csv")
    return np.genfromtxt(fname, delimiter=",", skip_header=True,
                         dtype=np.dtype(str))


@pytest.fixture(scope="session")
def titanic_sample():
    fname = os.path.join(TESTS_DIR, "titanic.csv")
    return load_config_sample(fname)


@pytest.fixture(scope="session")
def email_client():
    # Small example of the configuration of an email client presented in:
    # Saeideh Hamidi. Automating Software Customization via
    # Crowdsourcing using Association Rule Mining and Markov Decision
    # Processes. Master's thesis, York University, 2014.

    # Table 2 - Contingency Table of preferences in Email Client Example.
    # (The entry {disp=no,ico=lgi} was incorrectly labelled as 540
    # instead of 560 in the thesis. It was corrected in the CSV file.)
    fname = os.path.join(TESTS_DIR, "email_client.csv")
    config_sample = load_config_sample(fname)
    # {yes = 1, no = 0}, {smi = 1, lgi = 0}
    config_values = [[1, 0], [1, 0]]
    config = {0: 0, 1: 0}

    # Only one rule obtained with confidence 0.9 (page 46).
    min_supp, min_conf = 0.5, 0.9

    # The second question should be asked first. Then, if the user
    # answers ico=lgi it is possible to use the discovered association
    # rule to predict disp=no and only one question is needed.
    questions = [1]

    field_names = ["config_sample", "config_values", "min_support",
                   "min_confidence", "questions", "config"]
    EmailClient = namedtuple("EmailClient", field_names)
    email_client = EmailClient(config_sample=config_sample,
                               config_values=config_values,
                               min_support=min_supp,
                               min_confidence=min_conf,
                               questions=questions, config=config)
    return email_client


@pytest.fixture(scope="session")
def grid_world():
    # Example 4.1 of Reinforcement Learning: An Introduction
    # by Richard S. Sutton and Andrew G. Barto.

    S = 15  # {1, 2, ..., 14} and the terminal state 15
    initial_state, terminal_state = 11, 14
    A = 4  # {up, down, right, left}

    up, down, right, left = range(A)

    # Transitions.
    P = np.zeros((A, S, S))

    # Grid transitions.
    grid_transitions = {
        # from_state: ((action, to_state), ...)
        1: ((down, 5), (right, 2), (left, 15)),
        2: ((down, 6), (right, 3), (left, 1)),
        3: ((down, 7), (left, 2)),
        4: ((up, 15), (down, 8), (right, 5)),
        5: ((up, 1), (down, 9), (right, 6), (left, 4)),
        6: ((up, 2), (down, 10), (right, 7), (left, 5)),
        7: ((up, 3), (down, 11), (left, 6)),
        8: ((up, 4), (down, 12), (right, 9)),
        9: ((up, 5), (down, 13), (right, 10), (left, 8)),
        10: ((up, 6), (down, 14), (right, 11), (left, 9)),
        11: ((up, 7), (down, 15), (left, 10)),
        12: ((up, 8), (right, 13)),
        13: ((up, 9), (right, 14), (left, 12)),
        14: ((up, 10), (right, 15), (left, 13))
    }
    for i, moves in grid_transitions.items():
        for a, j in moves:
            P[a, i - 1, j - 1] = 1.0

    # Border transitions.
    for i in (1, 2, 3):
        P[up, i - 1, i - 1] = 1.0
    for i in (12, 13, 14):
        P[down, i - 1, i - 1] = 1.0
    for i in (3, 7, 11):
        P[right, i - 1, i - 1] = 1.0
    for i in (4, 8, 12):
        P[left, i - 1, i - 1] = 1.0

    # 15 should be an absorbing state.
    P[:, terminal_state, terminal_state] = 1.0

    # Rewards.
    R = -1 * np.ones((A, S, S))
    R[:, terminal_state, :] = 0

    # Discounting factor.
    gamma = 1.0

    # Optimal policy.
    policy = (left, left, down,
              up, up, up, down,
              up, up, down, down,
              up, right, right)

    field_names = ["num_states", "num_actions", "transitions",
                   "rewards", "initial_state", "terminal_state",
                   "discount_factor", "policy"]
    GridWorld = namedtuple("GridWorld", field_names)
    grid_world = GridWorld(num_states=S, num_actions=A, transitions=P,
                           rewards=R, initial_state=initial_state,
                           terminal_state=terminal_state,
                           discount_factor=gamma, policy=policy)
    return grid_world

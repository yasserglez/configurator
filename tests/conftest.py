import os
import sys
import random
import logging
from collections import namedtuple

import pytest
import numpy as np

from configurator.rules import Rule
from configurator.util import get_domain


TESTS_DIR = os.path.abspath(os.path.dirname(__file__))


@pytest.fixture(scope="function", autouse=True)
def random_seed():
    random_seed = 12345
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
def titanic_sample():
    # R's Titanic dataset. Contingency table expanded into explicit
    # form using the expand.table function from the epitools package.
    csv_file = os.path.join(TESTS_DIR, "titanic.csv")
    return np.genfromtxt(csv_file, delimiter=",", skip_header=True,
                         dtype=np.dtype(str))


@pytest.fixture(scope="session")
def email_client():
    # Small example of the configuration of an email client presented in:
    # Saeideh Hamidi. Automating Software Customization via
    # Crowdsourcing using Association Rule Mining and Markov Decision
    # Processes. Master's thesis, York University, 2014.

    # Table 2 - Contingency Table of preferences in Email Client Example.
    # (The entry {disp=no,ico=lgi} was incorrectly labelled as 540
    # instead of 560 in the thesis. It is corrected in the CSV file.)
    csv_file = os.path.join(TESTS_DIR, "email_client.csv")
    sample = np.genfromtxt(csv_file, delimiter=",", skip_header=True,
                           dtype=np.dtype(str))
    domain = get_domain(sample)

    # One rule obtained with confidence 0.9 (page 46).
    rules = [Rule({1: "lgi"}, {0: "no"})]
    constraints = [((1, 0), lambda _, x: x[0] != "lgi" or x[1] == "no")]

    # The second question should be asked first. Then, if the user
    # answers ico=lgi it is possible to use the rule to predict
    # disp=no and only one question is needed.
    questions = [1]

    fields = dict(domain=domain, rules=rules,
                  constraints=constraints, sample=sample,
                  questions=questions)
    EmailClient = namedtuple("EmailClient", fields.keys())
    email_client = EmailClient(**fields)
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

    fields = dict(num_states=S, num_actions=A, transitions=P,
                  rewards=R, initial_state=initial_state,
                  terminal_state=terminal_state,
                  discount_factor=gamma, policy=policy)
    GridWorld = namedtuple("GridWorld", fields.keys())
    grid_world = GridWorld(**fields)
    return grid_world

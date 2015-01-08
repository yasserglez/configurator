import os
from collections import defaultdict

import numpy as np


def _load_csv(csv_file):
    tests_dir = os.path.abspath(os.path.dirname(__file__))
    return np.genfromtxt(os.path.join(tests_dir, csv_file), delimiter=",",
                         skip_header=True, dtype=np.dtype(str))


def load_titanic():
    # R's Titanic dataset. Contingency table expanded into explicit
    # form using epitools' expand.table function.
    return _load_csv("titanic.csv")


def load_gridworld():
    # Example 4.1 of Reinforcement Learning: An Introduction
    # by Richard S. Sutton and Andrew G. Barto.

    S = 15  # {1, 2, ..., 14} and the terminal state 15
    INITIAL_S, TERMINAL_S = 11, 14
    A = 4  # {up, down, right, left}

    up, down, right, left = range(A)

    # Transitions.
    P = np.zeros((A, S, S))

    # Grid transitions.
    grid_transitions = {
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
        14: ((up, 10), (right, 15), (left, 13)),
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
    P[:, TERMINAL_S, TERMINAL_S] = 1.0

    # Rewards.
    R = -1 * np.ones((A, S, S))
    R[:, TERMINAL_S, :] = 0

    # Discounting factor.
    GAMMA = 1.0

    # Optimal policy.
    POLICY = {
        1: left,
        2: left,
        3: down,
        4: up,
        5: up,
        6: up,
        7: down,
        8: up,
        9: up,
        10: down,
        11: down,
        12: up,
        13: right,
        14: right,
    }

    return S, A, P, R, INITIAL_S, TERMINAL_S, GAMMA, POLICY


def load_email_client():
    # Small example of the configuration of an email client presented in:
    # Saeideh Hamidi. Automating Software Customization via
    # Crowdsourcing using Association Rule Mining and Markov Decision
    # Processes. Master's thesis, York University, 2014.

    # Table 2 - Contingency Table of preferences in Email Client Example.
    # (The entry {disp=no,ico=lgi} was incorrectly labelled as 540
    # instead of 560 in the thesis. It was corrected in the CSV file.)
    config_sample = _load_csv("email_client.csv")

    # Only one rule obtained with confidence 0.9 (page 46).
    min_support, min_confidence = 0.5, 0.9

    # The second question (ico) should be asked first.
    policy = defaultdict(lambda: 0)
    policy[frozenset()] = 1

    return config_sample, min_support, min_confidence, policy

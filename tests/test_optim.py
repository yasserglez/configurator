import random

from configurator.optim import PermutationDialog


def test_permutation_dialog():
    num_vars = 10
    config = {i: i % 2 for i in range(num_vars)}
    config_values = {i: [True, False] for i in range(num_vars)}
    var_perm = list(range(num_vars))
    random.shuffle(var_perm)
    dialog = PermutationDialog(config_values, [], var_perm, validate=True)
    dialog.reset()
    i = 0
    while not dialog.is_complete():
        var_index = dialog.get_next_question()
        assert var_index == var_perm[i]
        dialog.set_answer(var_index, config[var_index])
        i += 1
    assert dialog.config == config

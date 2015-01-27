import random

from ..sequence import SequenceDialog


def test_sequence_config_dialog():
    random.seed(42)
    num_vars = 10
    config = {i: i % 2 for i in range(num_vars)}
    config_values = {i: [True, False] for i in range(num_vars)}
    var_seq = list(range(num_vars))
    random.shuffle(var_seq)
    dialog = SequenceDialog(config_values, [], var_seq)
    dialog.reset()
    i = 0
    while not dialog.is_complete():
        var_index = dialog.get_next_question()
        assert var_index == var_seq[i]
        dialog.set_answer(var_index, config[var_index])
        i += 1
    assert dialog.config == config

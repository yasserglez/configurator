from configurator.dialogs import PermutationDialog
from configurator.util import get_domain, simulate_dialog


def test_get_domain(email_client):
    assert get_domain(email_client.sample) == email_client.domain


def test_simulate_dialog(email_client):
    dialog = PermutationDialog(email_client.domain, [1, 0],
                               constraints=email_client.constraints)
    for config, num_questions in email_client.scenarios:
        assert simulate_dialog(dialog, config) == num_questions

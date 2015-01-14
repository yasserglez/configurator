
from .examples import load_email_client
from ..policy import MDPDialogBuilder
from ..util import simulate_dialog, cross_validation


EMAIL_CLIENT = load_email_client()


def test_simulate_dialog():
    builder = MDPDialogBuilder(
        config_sample=EMAIL_CLIENT.config_sample,
        assoc_rule_min_support=EMAIL_CLIENT.min_support,
        assoc_rule_min_confidence=EMAIL_CLIENT.min_confidence)
    dialog = builder.build_dialog()
    accuracy, questions = simulate_dialog(dialog, EMAIL_CLIENT.config)
    assert accuracy == 1.0
    assert questions == 0.5

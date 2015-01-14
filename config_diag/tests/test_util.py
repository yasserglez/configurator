
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


def test_cross_validation():
    n_folds = 10
    random_state = 42
    builder_class = MDPDialogBuilder
    builder_kwargs = {"assoc_rule_min_support": EMAIL_CLIENT.min_support,
                      "assoc_rule_min_confidence": EMAIL_CLIENT.min_confidence}
    df = cross_validation(n_folds, random_state, builder_class,
                          builder_kwargs, EMAIL_CLIENT.config_sample)
    assert len(df.index) == n_folds
    assert ((0.5 <= df["accuracy_mean"]) & (df["accuracy_mean"] <= 1)).all()
    assert ((0 <= df["accuracy_std"]) & (df["accuracy_std"] <= 0.25)).all()
    assert (df["questions_mean"] == 0.5).all()
    assert (df["questions_std"] == 0).all()

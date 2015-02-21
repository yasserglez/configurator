from numpy.testing import assert_raises

from configurator.dialogs import DialogBuilder, Dialog


class TestDialogBuilder(object):

    def test_rules_constraints_one_required(self, email_client):
        assert_raises(ValueError, DialogBuilder, email_client.domains)

    def test_rules_constraints_mutually_exclusive(self, email_client):
        assert_raises(ValueError, DialogBuilder, email_client.domains,
                      email_client.rules, email_client.constraints)


class TestDialog(object):

    def test_reset(self, email_client):
        dialog = Dialog(email_client.domains, rules=email_client.rules)
        assert dialog.config == {}
        dialog.set_answer(0, "yes")
        assert dialog.config != {}
        dialog.reset()
        assert dialog.config == {}

    def test_set_answer_rules(self, email_client):
        dialog = Dialog(email_client.domains, rules=email_client.rules)
        dialog.set_answer(1, "lgi")
        assert dialog.is_complete()

    def test_set_answer_constraints(self, email_client):
        dialog = Dialog(email_client.domains,
                        constraints=email_client.constraints)
        dialog.set_answer(1, "lgi")
        assert dialog.is_complete()

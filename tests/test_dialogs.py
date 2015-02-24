from numpy.testing import assert_raises

from configurator.dialogs import DialogBuilder, Dialog, PermutationDialog


class TestDialogBuilder(object):

    def test_rules_constraints_one_required(self, email_client):
        assert_raises(ValueError, DialogBuilder, email_client.var_domains)

    def test_rules_constraints_mutually_exclusive(self, email_client):
        assert_raises(ValueError, DialogBuilder, email_client.var_domains,
                      email_client.rules, email_client.constraints)


class TestDialog(object):

    def test_reset(self, email_client):
        dialog = Dialog(email_client.var_domains, rules=email_client.rules,
                        validate=True)
        dialog.reset()
        assert dialog.config == {}
        dialog.set_answer(0, "yes")
        assert dialog.config != {}
        dialog.reset()
        assert dialog.config == {}

    def test_set_answer_rules(self, email_client):
        dialog = Dialog(email_client.var_domains, rules=email_client.rules,
                        validate=True)
        dialog.reset()
        dialog.set_answer(1, "lgi")
        assert dialog.is_complete()

    def test_set_answer_constraints(self, email_client):
        dialog = Dialog(email_client.var_domains,
                        constraints=email_client.constraints,
                        validate=True)
        dialog.reset()
        dialog.set_answer(1, "lgi")
        assert dialog.is_complete()


class TestPermutationDialog(object):

    def test_get_next_question_rules(self, email_client):
        dialog = PermutationDialog(email_client.var_domains, [1, 0],
                                   rules=email_client.rules,
                                   validate=True)
        dialog.reset()
        assert dialog.get_next_question() == 1
        dialog.set_answer(1, "lgi")
        assert dialog.is_complete()

    def test_get_next_question_constraints(self, email_client):
        dialog = PermutationDialog(email_client.var_domains, [1, 0],
                                   constraints=email_client.constraints,
                                   validate=True)
        dialog.reset()
        assert dialog.get_next_question() == 1
        dialog.set_answer(1, "lgi")
        assert dialog.is_complete()

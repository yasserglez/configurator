from configurator.sequence import PermutationDialog


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

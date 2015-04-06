import pytest

from configurator.sequence import PermutationDialog


class TestPermutationDialog(object):

    @pytest.mark.parametrize("use_rules", (True, False),
                             ids=("rules", "constraints"))
    def test_get_next_question(self, use_rules, email_client):
        if use_rules:
            rules = email_client.rules
            constraints = None
        else:
            rules = None
            constraints = email_client.constraints
        dialog = PermutationDialog(email_client.var_domains, [1, 0],
                                   rules=rules,
                                   constraints=constraints,
                                   validate=True)
        dialog.reset()
        assert dialog.get_next_question() == 1
        dialog.set_answer(1, "lgi")
        assert dialog.is_complete()

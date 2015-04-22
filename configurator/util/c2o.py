#
#  Copyright 2015 Yasser Gonzalez Fernandez
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#

import itertools
from collections import defaultdict

from bs4 import BeautifulSoup


def load_C2O(xml_file):
    """Load a C2O decision model.

    See http://www.jku.at/isse/content/e139529/e126342/e126343 for
    information about the C2O (Configurator 2.0) tool.

    Arguments:
        xml_file: Path to a C2O XML file.

    Returns:
        A tuple with two elements `var_domains` and `constraints`
        giving the variable domains and constraints, respectively.
    """

    with open(xml_file) as fd:
        soup = BeautifulSoup(fd, "xml")

    # Fail early if the file contains cnfRelation tags.
    if soup.find("cnfRelation"):
        raise ValueError("CNF relations not yet supported")

    # Parse the questions.
    # Keys: a question id.
    # Values: set of possible answers.
    questions = {}
    for question in soup.model.questions.find_all("question"):
        choices = set()
        for choice in question.find_all("choice"):
            choices.add(choice["name"])
        questions[question["identifier"]] = choices

    # Parse the constraint relations.
    # Keys: (source question, target question) tuples.
    # Values: dict mapping a source choice to a set of allowed target choices.
    restrictions = defaultdict(dict)
    for constraint in soup.model.relations.find_all("constraintRelation"):
        source_id = constraint.source["questionIdentifier"]
        for rule in [r for r in constraint.find_all("rule") if r.contents]:
            source_choice = rule["choiceName"]
            for target in constraint.targets.find_all("target"):
                target_id = target["questionIdentifier"]
                target_choices = set()
                if rule.contents[0].name == "allowed":
                    for allowed in rule.find_all("allowed"):
                        target_choices.add(allowed["choiceName"])
                else:
                    target_choices.update(questions[target_id])
                    for disallowed in rule.find_all("disallowed"):
                        target_choices.remove(disallowed["choiceName"])
                # Populate the constraints dict.
                k = (source_id, target_id)
                restrictions[k][source_choice] = target_choices

    # Parse the relevancy relations.
    # Keys: (source question, target question) tuples.
    # Values: a set of choices of the source question that makes
    #         the target question irrelevant.
    relevancies = {}
    for relevancy in soup.model.relations.find_all("relevancyRelation"):
        source_id = relevancy.source["questionIdentifier"]
        for target in relevancy.targets.find_all("target"):
            target_id = target["questionIdentifier"]
            irrelevant_choices = set()
            if any(t.name == "irrelevantIf" for t in relevancy.find_all(True)):
                for irrelevant in relevancy.find_all("irrelevantIf"):
                    irrelevant_choices.add(irrelevant["choiceName"])
            else:
                irrelevant_choices.update(questions[source_id])
                for relevant in relevancy.find_all("relevantIf"):
                    irrelevant_choices.remove(relevant["choiceName"])
            # Populate the relevancies dict.
            k = (source_id, target_id)
            relevancies[k] = irrelevant_choices

    # Transform the problem into the representation used in the library.

    irrelevant_value = "Irrelevant"

    question_ids = {i: question for i, question in
                    enumerate(sorted(questions.keys()))}
    question_indices = {q: i for i, q in question_ids.items()}
    var_domains = [list(sorted(questions[question]))
                   for question in sorted(questions.keys())]
    for question, i in question_indices.items():
        assert questions[question] == set(var_domains[i])
        assert irrelevant_value not in questions[question]

    # Add an additional value for variables that can become irrelevant.
    for source, target in relevancies.keys():
        var_domain = var_domains[question_indices[target]]
        if irrelevant_value not in var_domain:
            var_domain.append(irrelevant_value)

    def constraint_fun(var_indices, var_values):
        source = question_ids[var_indices[0]]
        target = question_ids[var_indices[1]]
        source_value, target_value = var_values
        # Check for restrictions.
        if (source, target) in restrictions:
            allowed_target_choices = restrictions[(source, target)]
            if (source_value in allowed_target_choices and
                    target_value not in allowed_target_choices[source_value]):
                return False
        # Check for relevancy relations.
        if (source, target) in relevancies:
            irrelevant_choices = relevancies[(source, target)]
            if (source_value in irrelevant_choices and
                    target_value != irrelevant_value):
                return False
        # Passed all the constraints.
        return True

    constraints = {}
    for source, target in \
            itertools.chain(restrictions.keys(), relevancies.keys()):
        if (source, target) not in constraints:
            var_indices = (question_indices[source],
                           question_indices[target])
            constraints[var_indices] = constraint_fun
    constraints = list(constraints.items())

    return var_domains, constraints

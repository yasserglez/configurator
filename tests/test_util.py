import os

from conftest import TESTS_DIR

from configurator.util import load_C2O
from configurator.csp import CSP


def test_load_C2O():
    # http://www.jku.at/JKU_Site/JKU/isse/content/e139529/e126342/e126343/e266488/Car.xml
    xml_file = os.path.join(TESTS_DIR, "Car.xml")
    var_domains, constraints = load_C2O(xml_file)
    csp = CSP(var_domains, constraints)
    num_solutions = 0
    for solution in csp.solve():
        num_solutions += 1
    assert num_solutions == 168

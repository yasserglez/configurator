import re
import ast
import collections

from bs4 import BeautifulSoup


# The SXFM (Simple XML Feature Model) format documentation is
# available at http://gsd.uwaterloo.ca:8088/SPLOT/sxfm.html.

def load_SXFM(xml_file):
    with open(xml_file) as f:
        soup = BeautifulSoup(f, 'xml')

    # Parse the feature tree.
    feature_tree_str = soup.find('feature_tree').get_text()
    var_indices, root_node = _parse_feature_tree(feature_tree_str.strip())

    # Variable domains.
    var_domains = [[True, False] for i in range(len(var_indices))]
    var_domains[var_indices[root_node.node_id]] = [True]

    constraints_index = collections.defaultdict(list)

    # Add the tree constraints.
    node_stack = [root_node]
    while node_stack:
        node = node_stack.pop()
        for child_node in node.children:
            node_stack.append(child_node)
        # Add the corresponding node constraint.
        if node.node_type == 'm':
            fp = var_indices[node.parent.node_id]
            fc = var_indices[node.node_id]
            constraint_vars = [fp, fc]
            constraint = [constraint_vars, _mandatory_constraint_fun]
            constraint_key = frozenset(constraint_vars)
            constraints_index[constraint_key].append(constraint)
        elif node.node_type == 'o':
            fp = var_indices[node.parent.node_id]
            fc = var_indices[node.node_id]
            constraint_vars = [fp, fc]
            constraint = [constraint_vars, _optional_constraint_fun]
            constraint_key = frozenset(constraint_vars)
            constraints_index[constraint_key].append(constraint)
        elif node.node_type == 'g':
            assert node.min_card == 1
            assert node.max_card in {1, -1}
            constraint_vars = [var_indices[node.parent.node_id]]
            for child_node in node.children:
                constraint_vars.append(var_indices[child_node.node_id])
            if node.max_card == 1:
                constraint = [constraint_vars, _xor_constraint_fun]
            else:
                constraint = [constraint_vars, _or_constraint_fun]
            constraint_key = frozenset(constraint_vars)
            constraints_index[constraint_key].append(constraint)

    # Add cross-tree constraints.
    constraints_str = soup.find('constraints').get_text()
    for constraint in _parse_constraints(var_indices, constraints_str.strip()):
        constraint_key = frozenset(constraint[0])
        constraints_index[constraint_key].append(constraint)

    # Normalize the constraints.
    normalized_constraints = []
    for constraints in constraints_index.values():
        if len(constraints) == 1:
            normalized_constraints.append(constraints[0])
        else:
            constraint_fun = _normalize_constraints(constraints)
            normalized_constraints.append([constraints[0][0], constraint_fun])
    return var_domains, normalized_constraints


def _normalize_constraints(constraints):
    def _normalized_constraint_fun(var_indices, var_values):
        for constraint_var_indices, constraint_fun in constraints:
            constraint_var_values = [var_values[i] for i in
                                     (var_indices.index(var_index)
                                      for var_index in constraint_var_indices)]
            if not constraint_fun(constraint_var_indices,
                                  constraint_var_values):
                return False
        return True
    return _normalized_constraint_fun


def _mandatory_constraint_fun(var_indices, var_values):
    fp, fc = var_values
    return fp == fc


def _optional_constraint_fun(var_indices, var_values):
    fp, fc = var_values
    return (not fc) or fp


def _xor_constraint_fun(var_indices, var_values):
    fp, fc = var_values[0], var_values[1:]
    if fp:
        return sum(fc) == 1
    else:
        return sum(fc) == 0


def _or_constraint_fun(var_indices, var_values):
    fp, fc = var_values[0], var_values[1:]
    if fp:
        return sum(fc) > 0
    else:
        return sum(fc) == 0


def _parse_feature_tree(feature_tree_str):
    var_indices = {}
    node_stack = []
    for node_str in feature_tree_str.splitlines():
        node_level = len(re.match(r'^\t*', node_str).group(0))
        node = _parse_node(node_str.strip())
        if node.node_type != 'g':
            if node.node_id is None:
                node.node_id = '_id_{}'.format(len(var_indices))
            assert node.node_id not in var_indices
            var_indices[node.node_id] = len(var_indices)
        if not node_stack:
            # The stack is empty, it must be the root node.
            assert node.node_type == 'r' and node_level == 0
            node_stack.append(node)
        else:
            current_level = len(node_stack) - 1
            if node_level > current_level:
                # It is a child node.
                node_stack[-1].add_child(node)
                node_stack.append(node)
            elif node_level == current_level:
                # Another node at the same level. We keep the last
                # children at the top of the stack.
                node_stack.pop()
                node_stack[-1].add_child(node)
                node_stack.append(node)
            else:
                # It is another branch of a previous node.
                num_pops = current_level - node_level + 1
                for i in range(num_pops):
                    node_stack.pop()
                node_stack[-1].add_child(node)
                node_stack.append(node)
    return var_indices, node_stack[0]


class _TreeNode(object):

    def __init__(self, node_type, node_id, min_card=None, max_card=None):
        super().__init__()
        self.node_type = node_type
        self.node_id = node_id
        self.min_card = min_card
        self.max_card = max_card
        self.parent = None
        self.children = []

    def add_child(self, node):
        self.children.append(node)
        node.parent = self

    def __str__(self):
        return 'TreeNode(node_type=%r, node_id=%r, min_card=%r, max_card=%r)' % \
            (self.node_type, self.node_id, self.min_card, self.max_card)


_rmo_re = re.compile(r':(?P<node_type>[rmo])[^(]+(\((?P<node_id>\w+)\))?')

_g_re = re.compile(r':g.*?\[(?P<min_card>\d+),(?P<max_card>\d+|\*)\]$')

_other_re = re.compile(r':.*?(\((?P<node_id>\w+)\))?$')

def _parse_node(node_str):
    if node_str[1] in 'rmo':
        match = _rmo_re.match(node_str)
        node = _TreeNode(match.group('node_type'), match.group('node_id'))
    elif node_str[1] in 'g':
        match = _g_re.match(node_str)
        min_card = int(match.group('min_card'))
        max_card = (-1 if match.group('max_card') == '*' else
                    int(match.group('max_card')))
        node = _TreeNode('g', None, min_card, max_card)
    else:
        match = _other_re.match(node_str)
        node = _TreeNode('', match.group('node_id'))
    return node


class _CollectNames(ast.NodeVisitor):

    accepted_nodes = {ast.Expression, ast.Name,
                      ast.BoolOp, ast.Or, ast.And,
                      ast.UnaryOp, ast.Not}

    def __init__(self):
        super().__init__()
        self.var_names = []

    def visit_Name(self, node):
        self.var_names.append(node.id)
        return node

    def visit(self, node):
        if type(node) not in self.accepted_nodes:
            raise SyntaxError("Invalid CNF expression")
        return super().visit(node)


class _RewriteName(ast.NodeTransformer):

    def __init__(self, var_names):
        super().__init__()
        self.var_names = var_names

    def visit_Name(self, node):
        return ast.Subscript(
            value=ast.Name(id='var_values', ctx=ast.Load()),
            slice=ast.Index(value=ast.Num(n=self.var_names.index(node.id))),
            ctx=node.ctx)


def _parse_constraints(var_indices, constraints_str):
    for constraint_str in constraints_str.splitlines():
        constraint_str = constraint_str[constraint_str.find(':') + 1:].strip()
        constraint_str = constraint_str.replace('~', 'not ')
        cnf_expr = ast.parse(constraint_str, '<string>', 'eval')
        # Substitute the variable names by the indices in the CNF expression.
        collector = _CollectNames()
        collector.visit(cnf_expr)
        rewriter = _RewriteName(collector.var_names)
        cnf_expr = ast.fix_missing_locations(rewriter.visit(cnf_expr))
        compiled_cnf_expr = compile(cnf_expr, constraint_str, 'eval')
        # Yield the constraint function.
        constraint_vars = [var_indices[var_name]
                           for var_name in collector.var_names]
        def constraint_fun(compiled_cnf_expr):
            return lambda var_indices, var_values: \
                eval(compiled_cnf_expr,
                     {'__builtins__': {}},
                     {'var_values': var_values})
        yield constraint_vars, constraint_fun(compiled_cnf_expr)

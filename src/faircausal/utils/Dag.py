import networkx as nx


def is_valid_causal_dag(dag: dict):
    """
    Check if the given graph is a valid DAG.

    :param dag: Directed Acyclic Graph (DAG) represented as a dictionary
    :return: True if the graph is a valid DAG, False otherwise
    """
    for node, children in dag.items():
        for child in children:
            if child not in dag:
                raise ValueError(f"Node {child} is not present in the graph.")

    def has_cycle(dag: dict):
        """
        Check if the given graph has a cycle.

        :param dag: Directed Acyclic Graph (DAG) represented as a dictionary
        :return: True if the graph has a cycle, False otherwise
        """
        visited = set()
        stack = set()

        def visit(node):
            if node in stack:
                return True
            if node in visited:
                return False
            visited.add(node)
            stack.add(node)
            for child in dag.get(node, []):
                if visit(child):
                    return True
            stack.remove(node)
            return False

        return any(visit(node) for node in dag)

    if has_cycle(dag):
        return False

    return True


def find_parents(causal_dag: dict, variable: str):
    """
    Find the parents of a given variable in the causal DAG.

    :param causal_dag: Dictionary representing the causal DAG.
    :param variable: The variable for which to find the parents.
    :return: List of parent variables.
    """
    return [node for node, children in causal_dag.items() if variable in children]


def classify_confounders_mediators(causal_dag: dict, exposure: str, outcome: str):
    """
    Classify the confounders and mediators in the causal DAG.

    :param causal_dag: Dictionary representing the causal DAG.
    :param exposure: The exposure variable.
    :param outcome: The outcome variable.
    :return: Dictionary with keys 'mediators' and 'confounders' containing the mediators and confounders respectively.
    """
    dag = nx.DiGraph()
    for node, children in causal_dag.items():
        for child in children:
            dag.add_edge(node, child)

    # show all paths from exposure to outcome
    paths = list(nx.all_simple_paths(dag, source=exposure, target=outcome))

    # find mediators
    mediators = set()
    for path in paths:
        for node in path:
            if node != exposure and node != outcome:
                mediators.add(node)

    # find confounders
    confounders = set()
    for node in dag.nodes:
        if node != exposure and node != outcome:
            if nx.has_path(dag, node, exposure) and nx.has_path(dag, node, outcome):
                if node not in mediators:
                    confounders.add(node)

    return {
        "mediators": list(mediators),
        "confounders": list(confounders)
    }

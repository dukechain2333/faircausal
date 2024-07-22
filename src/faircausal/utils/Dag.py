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

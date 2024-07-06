def is_valid_causal_dag(dag: dict):
    for node, children in dag.items():
        for child in children:
            if child not in dag:
                raise ValueError(f"Node {child} is not present in the graph.")

    def has_cycle(dag: dict):
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

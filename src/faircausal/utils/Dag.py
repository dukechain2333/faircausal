import networkx as nx
import pandas as pd
from sklearn.linear_model import LogisticRegression


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


# def is_connected(dag: dict):
#     """
#     Check if all nodes in the graph are connected.
#
#     :param dag: Graph represented as a dictionary
#     :return: True if all nodes are connected, False otherwise
#     """
#     # Empty graph is considered connected
#     if not dag:
#         return True
#
#     def dfs(node, visited):
#         visited.add(node)
#         for neighbor in dag.get(node, []):
#             if neighbor not in visited:
#                 dfs(neighbor, visited)
#
#     # Get all nodes in the graph
#     all_nodes = set(dag.keys()).union(*dag.values())
#
#     # Start DFS from the first node
#     start_node = next(iter(all_nodes))
#     visited = set()
#     dfs(start_node, visited)
#
#     # Check if all nodes were visited
#     return len(visited) == len(all_nodes)

def is_connected(dag: dict):
    """
    Check if all nodes in the graph have at least one incoming or outgoing edge.

    :param dag: Graph represented as a dictionary
    :return: True if all nodes are connected (have either incoming or outgoing edges), False otherwise
    """
    # Empty graph is considered connected
    if not dag:
        return True

    # Get all nodes in the graph
    all_nodes = set(dag.keys()).union(*dag.values())

    # Check if each node has either an incoming or outgoing edge
    for node in all_nodes:
        # Check if the node is either a key (outgoing edge) or a value (incoming edge)
        if node not in dag or (node in dag and len(dag[node]) == 0):
            has_incoming = any(node in neighbors for neighbors in dag.values())
            if not has_incoming:
                return False

    return True


def is_valid_causal_dag(dag: dict):
    """
    Check if the given graph is a valid DAG, i.e., it has no cycles and all nodes are connected.

    :param dag: Directed Acyclic Graph (DAG) represented as a dictionary
    :return: True if the graph is a valid DAG with all nodes connected, False otherwise
    """
    # Check if all child nodes are present in the graph
    for node, children in dag.items():
        for child in children:
            if child not in dag:
                raise ValueError(f"Node {child} is not present in the graph.")

    if has_cycle(dag):
        return False

    if not is_connected(dag):
        return False

    return True


def remove_edges(graph_dict, edges):
    """
    Remove specified edge(s) from the graph dictionary.

    :param graph_dict: Dictionary representation of the graph
    :param edges: A single edge as a tuple (start_node, end_node) or a list of such tuples
    :return: The modified graph dictionary
    :raises ValueError: If any specified edge doesn't exist in the graph
    """
    if isinstance(edges, tuple):
        edges = [edges]
    elif not isinstance(edges, list):
        raise ValueError("Input must be a single edge tuple or a list of edge tuples")

    for edge in edges:
        if not isinstance(edge, tuple) or len(edge) != 2:
            raise ValueError(f"Invalid edge format: {edge}. Must be a tuple of two elements.")

        start_node, end_node = edge

        if start_node not in graph_dict:
            raise ValueError(f"Start node '{start_node}' not found in the graph")

        if end_node not in graph_dict:
            raise ValueError(f"End node '{end_node}' not found in the graph")

        if end_node not in graph_dict[start_node]:
            raise ValueError(f"Edge from '{start_node}' to '{end_node}' does not exist in the graph")

        graph_dict[start_node].remove(end_node)

    return graph_dict


def remove_nodes(graph_dict, nodes):
    """
    Remove specified node(s) and all related edges from the graph dictionary.

    :param graph_dict: Dictionary representation of the graph
    :param nodes: A single node or a list of nodes to be removed
    :return: The modified graph dictionary
    :raises ValueError: If any specified node doesn't exist in the graph
    """
    if isinstance(nodes, (str, int)):  # If a single node is provided
        nodes = [nodes]
    elif not isinstance(nodes, list):
        raise ValueError("Input must be a single node or a list of nodes")

    for node in nodes:
        if node not in graph_dict:
            raise ValueError(f"Node '{node}' not found in the graph")

    for node in nodes:
        # Remove the node and its outgoing edges
        del graph_dict[node]

        # Remove incoming edges to this node
        for remaining_node in graph_dict:
            if node in graph_dict[remaining_node]:
                graph_dict[remaining_node].remove(node)

    return graph_dict


def add_node(graph_dict, node, children=None, parents=None):
    """
    Add a single node to the graph dictionary, along with its children and parents.

    :param graph_dict: Dictionary representation of the graph
    :param node: The node to be added
    :param children: A list of child nodes for the new node
    :param parents: A list of parent nodes for the new node
    :return: The modified graph dictionary
    :raises ValueError: If the specified node already exists in the graph
    """
    if node in graph_dict:
        raise ValueError(f"Node '{node}' already exists in the graph")

    children = children or []
    parents = parents or []

    # Add the new node with its children
    graph_dict[node] = children

    # Add edges from parents to the new node
    for parent in parents:
        if parent not in graph_dict:
            raise ValueError(f"Parent node '{parent}' not found in the graph")
        graph_dict[parent].append(node)

    # Ensure all children exist in the graph
    for child in children:
        if child not in graph_dict:
            graph_dict[child] = []

    return graph_dict


def add_edges(graph_dict, edges):
    """
    Add edge(s) to the graph dictionary.

    :param graph_dict: Dictionary representation of the graph
    :param edges: A single edge as a tuple (start_node, end_node) or a list of such tuples
    :return: The modified graph dictionary
    :raises ValueError: If the input format is incorrect
    """
    if isinstance(edges, tuple):
        edges = [edges]
    elif not isinstance(edges, list):
        raise ValueError("Edges must be a single tuple (start_node, end_node) or a list of such tuples")

    for edge in edges:
        if not isinstance(edge, tuple) or len(edge) != 2:
            raise ValueError(f"Invalid edge format: {edge}. Must be a tuple of two elements.")

        start_node, end_node = edge

        # If the start node doesn't exist, create it
        if start_node not in graph_dict:
            graph_dict[start_node] = []

        # If the end node doesn't exist, create it
        if end_node not in graph_dict:
            graph_dict[end_node] = []

        # Add the edge if it doesn't already exist
        if end_node not in graph_dict[start_node]:
            graph_dict[start_node].append(end_node)

    return graph_dict



def remove_unconnected_nodes(dag: dict, target_node: str = None):
    """
    Remove all unconnected nodes from the given DAG, and optionally retain only the part of the DAG
    containing a specified root node.

    :param dag: Directed Acyclic Graph (DAG) represented as a dictionary
    :param target_node: The node to use as the root for the sub-DAG to retain. If None, all connected components are considered.
    :return: A new DAG dictionary with unconnected nodes removed and optionally only the sub-DAG containing the root_node.
    """

    def find_connected_nodes(dag: dict, start_node: str = None):
        """
        Identify all connected nodes in the graph, optionally starting from a specific node.

        :param dag: Directed Acyclic Graph (DAG) represented as a dictionary
        :param start_node: The node to start DFS from. If None, all nodes are considered.
        :return: A set of connected nodes
        """
        connected_nodes = set()

        def dfs(node):
            if node not in connected_nodes:
                connected_nodes.add(node)
                for child in dag.get(node, []):
                    dfs(child)

        # If a start_node is provided, only traverse the graph from that node
        if start_node:
            dfs(start_node)
        else:
            # Start DFS from every node to gather all connected nodes
            for node in dag.keys():
                dfs(node)

        # Nodes that are either parents or children
        all_involved_nodes = set(dag.keys()) | {child for children in dag.values() for child in children}
        return connected_nodes & all_involved_nodes

    # Check if the root_node exists and is connected
    if target_node and target_node not in dag:
        raise ValueError(f"Node {target_node} is not present in the DAG.")

    # Get the set of connected nodes, optionally starting from the root_node
    connected_nodes = find_connected_nodes(dag, start_node=target_node)

    # If the root_node is specified but has no connections, return an empty DAG
    if target_node and target_node not in connected_nodes:
        raise Warning(f"Node {target_node} has no connections in the DAG.")

    # Filter the original DAG to keep only the connected nodes
    new_dag = {node: [child for child in children if child in connected_nodes]
               for node, children in dag.items() if node in connected_nodes}

    return new_dag


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


def recursive_predict(node: str, causal_dag: dict, linear_model: dict, data: pd.DataFrame, predicted_data=None,
                      final_predict_proba=False):
    """
    Recursively predict the value of a node in the causal DAG.

    :param node: The node to predict.
    :param causal_dag: Dictionary representing the causal DAG.
    :param linear_model: Dictionary containing linear models for each node in the DAG.
    :param data: DataFrame containing the data.
    :param predicted_data: Dictionary containing predicted values for each node.
    :param final_predict_proba: Whether to predict probabilities for classification tasks.
    :return: Predicted value for the node.
    """
    # Initialize predicted_data if it is None
    if predicted_data is None:
        predicted_data = {}

    # If the node is already predicted, return the value
    if node in predicted_data:
        return predicted_data[node]

    # Find the parents of the current node
    parents = find_parents(causal_dag, node)

    # Recursively predict the parent nodes
    if parents:
        for parent in parents:
            if parent not in predicted_data:
                predicted_data[parent] = recursive_predict(parent, causal_dag, linear_model, data, predicted_data)

    # Check if the node has a linear model
    if node in linear_model:
        model = linear_model[node]

        # Use the parent nodes' data to predict the current node
        if parents:
            X = pd.DataFrame({parent: predicted_data[parent] for parent in parents})
        else:
            X = data[parents]

        # Predict the value
        if isinstance(model, LogisticRegression) and final_predict_proba:
            predicted_value = model.predict_proba(X)
        else:
            predicted_value = model.predict(X)
    else:
        # If no model exists for the node, return the original data for the node
        predicted_value = data[node]

    # Store the predicted value
    predicted_data[node] = predicted_value

    return predicted_value

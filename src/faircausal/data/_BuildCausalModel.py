import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from causalnex.structure.notears import from_pandas

from faircausal.utils.Dag import find_parents


def build_causal_model(data: pd.DataFrame, max_iter: int = 100, w_threshold: float = 0.0):
    """
    Build a causal model from a DataFrame.
    :param data: DataFrame containing the data.
    :param max_iter: Maximum number of iterations for the optimization algorithm.
    :param w_threshold: Threshold for edge weights.
    :return: Causal model, DAG dictionary
    """

    # Build causal model
    s_model = from_pandas(data, max_iter=max_iter, w_threshold=w_threshold)
    dag_dict = {node: list(s_model.successors(node)) for node in s_model.nodes}

    return s_model, dag_dict


def generate_linear_models(dag: dict, data: pd.DataFrame):
    """
    Generate linear models for each node in the DAG.

    :param dag: Dictionary representing the causal DAG.
    :param data: DataFrame containing the data.
    :return: Dictionary containing linear models for each node in the DAG.
    """
    linear_models = {}

    for node in dag.keys():
        parents = find_parents(dag, node)
        if not parents:
            continue

        if data[node].dtype.name == 'category':
            model = LogisticRegression().fit(data[parents], data[node])
        else:
            model = LinearRegression().fit(data[parents], data[node])

        linear_models[node] = model

    return linear_models

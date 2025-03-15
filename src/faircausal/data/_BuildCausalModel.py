import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from faircausal.utils.Dag import find_parents


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
            model = LogisticRegression(solver='saga', max_iter=1000).fit(data[parents], data[node])
        else:
            model = LinearRegression().fit(data[parents], data[node])

        linear_models[node] = model

    return linear_models

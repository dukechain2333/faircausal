import pandas as pd
import statsmodels.api as sm
from causalnex.structure.notears import from_pandas


def build_causal_model(dataframe, max_iter: int = 100, w_threshold: float = 0.0):
    s_model = from_pandas(dataframe, max_iter=max_iter, w_threshold=w_threshold)
    dag_dict = {node: list(s_model.successors(node)) for node in s_model.nodes}
    return s_model, dag_dict


def generate_beta_params(dag: dict, data: pd.DataFrame):
    beta_params = {}

    parents_dict = {node: [] for node in dag}
    for parent, children in dag.items():
        for child in children:
            parents_dict[child].append(parent)

    for node, parents in parents_dict.items():
        if not parents:
            continue

        X = sm.add_constant(data[parents])
        model = sm.OLS(data[node], X).fit()
        beta_params[f'beta_{node}_0'] = model.params['const']
        for parent in parents:
            beta_params[f'beta_{node}_{parent}'] = model.params[parent]

    return beta_params

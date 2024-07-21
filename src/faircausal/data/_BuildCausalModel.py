import pandas as pd
import statsmodels.api as sm
from causalnex.structure.notears import from_pandas


def build_causal_model(data: pd.DataFrame, data_type: dict, max_iter: int = 100, w_threshold: float = 0.0):
    # One-hot encode discrete variables
    discrete_vars = [var for var, var_type in data_type.items() if var_type == 'discrete']
    data_encoded = pd.get_dummies(data, columns=discrete_vars, drop_first=False)

    # Update data_type with new one-hot encoded variables
    new_data_type = {}
    for var in data.columns:
        if var in discrete_vars:
            for dummy_var in data_encoded.columns:
                if dummy_var.startswith(var + "_"):
                    new_data_type[dummy_var] = 'discrete'
        else:
            new_data_type[var] = data_type[var]

    # Build causal model
    s_model = from_pandas(data_encoded, max_iter=max_iter, w_threshold=w_threshold)
    dag_dict = {node: list(s_model.successors(node)) for node in s_model.nodes}

    return s_model, dag_dict, data_encoded, new_data_type


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
        beta_params[f'beta__{node}__0'] = model.params['const']
        for parent in parents:
            beta_params[f'beta__{node}__{parent}'] = model.params[parent]

    return beta_params


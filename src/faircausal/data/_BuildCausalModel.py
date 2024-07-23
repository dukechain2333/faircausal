import pandas as pd
import statsmodels.api as sm
from causalnex.structure.notears import from_pandas

from faircausal.utils.Dag import find_parents


def build_causal_model(data: pd.DataFrame, data_type: dict, outcome_variable: str, max_iter: int = 100, w_threshold: float = 0.0):
    """
    Build a causal model from a DataFrame.

    :param data: DataFrame containing the data.
    :param data_type: Dictionary with variable names as keys and classifications as values.
    :param outcome_variable: The outcome variable which should not be dummy encoded.
    :param max_iter: Maximum number of iterations for the NOTEARS algorithm.
    :param w_threshold: Threshold for edge weights in the causal model.

    :return: Tuple containing the causal model, the DAG dictionary, the encoded data, and the new data type.
    """
    # One-hot encode discrete variables except the outcome variable
    discrete_vars = [var for var, var_type in data_type.items() if var_type == 'discrete' and var != outcome_variable]
    data_encoded = pd.get_dummies(data, columns=discrete_vars, drop_first=False)

    # Convert outcome_variable to numeric if it is a string (category)
    if data[outcome_variable].dtype == 'object':
        data_encoded[outcome_variable] = pd.factorize(data[outcome_variable])[0]
    else:
        data_encoded[outcome_variable] = data[outcome_variable]

    # Update data_type with new one-hot encoded variables
    new_data_type = {}
    for var in data.columns:
        if var in discrete_vars:
            for dummy_var in data_encoded.columns:
                if dummy_var.startswith(var + "_"):
                    new_data_type[dummy_var] = 'discrete'
        else:
            new_data_type[var] = data_type[var]

    # Ensure outcome_variable is in the new data_type
    if outcome_variable not in new_data_type:
        new_data_type[outcome_variable] = data_type[outcome_variable]

    # Build causal model
    s_model = from_pandas(data_encoded, max_iter=max_iter, w_threshold=w_threshold)
    dag_dict = {node: list(s_model.successors(node)) for node in s_model.nodes}

    return s_model, dag_dict, data_encoded, new_data_type


def generate_beta_params(dag: dict, data: pd.DataFrame):
    """
    Generate beta parameters for the causal model.

    :param dag: Dictionary representing the causal DAG.
    :param data: DataFrame containing the data.
    :return: Dictionary of beta parameters.
    """
    beta_params = {}

    for node in dag.keys():
        parents = find_parents(dag, node)
        if not parents:
            continue

        X = sm.add_constant(data[parents])
        model = sm.OLS(data[node], X).fit()
        beta_params[f'beta__{node}__0'] = model.params['const']
        for parent in parents:
            beta_params[f'beta__{node}__{parent}'] = model.params[parent]

    return beta_params

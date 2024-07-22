import numpy as np
import statsmodels.api as sm

from faircausal.data.CausalDataReader import CausalDataReader
from faircausal.utils.Dag import find_parents


def negative_log_likelihood(causal_data: CausalDataReader):
    """
    Negative Log-Likelihood for the causal model.

    This function assumes that all continuous variables are normally distributed and all discrete variables are
    dummy-encoded which follow a Binomial distribution.

    :param causal_data: CausalDataReader object
    :return: Negative Log-Likelihood
    """
    data = causal_data['data']
    data_type = causal_data['data_type']
    beta_dict = causal_data['beta_dict']
    causal_dag = causal_data['causal_dag']

    nll_continuous = 0
    nll_discrete = 0

    for node, dtype in data_type.items():
        parents = find_parents(causal_dag, node)
        if not parents:
            continue

        X = sm.add_constant(data[parents])
        y = data[node]

        betas = [beta_dict[f'beta__{node}__0']] + [beta_dict[f'beta__{node}__{parent}'] for parent in parents]
        y_pred = np.dot(X, betas)

        if dtype == 'continuous':
            sigma = np.std(y - y_pred)
            nll_continuous += np.sum(0.5 * np.log(2 * np.pi * sigma ** 2) + ((y - y_pred) ** 2) / (2 * sigma ** 2))

        elif dtype == 'discrete':
            n = 1
            p = 1 / (1 + np.exp(-y_pred))
            nll_discrete += -np.sum(y * np.log(p) + (n - y) * np.log(1 - p))

    total_nll = nll_continuous + nll_discrete
    return total_nll


def mse(causal_data: CausalDataReader, outcome_variable: str):
    """
    Mean Squared Error for the predicted outcome variable.

    :param causal_data: CausalDataReader object
    :param outcome_variable: The outcome variable for which to calculate MSE.
    :return: Mean Squared Error
    """
    if outcome_variable not in causal_data['data'].columns:
        raise ValueError(f"Outcome variable {outcome_variable} not found in data.")

    model_data = causal_data.get_model()
    data = model_data['data']
    beta_dict = model_data['beta_dict']
    causal_dag = model_data['causal_dag']

    parents = find_parents(causal_dag, outcome_variable)
    if not parents:
        raise ValueError(f"No parents found for outcome variable {outcome_variable} in causal DAG.")

    X = sm.add_constant(data[parents])
    y_true = data[outcome_variable]
    betas = [beta_dict[f'beta__{outcome_variable}__0']] + [beta_dict[f'beta__{outcome_variable}__{parent}'] for parent in parents]
    y_pred = np.dot(X, betas)
    mse_value = np.mean((y_true - y_pred) ** 2)

    return mse_value


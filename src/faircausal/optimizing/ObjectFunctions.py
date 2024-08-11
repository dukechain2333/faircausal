import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

from faircausal.data.CausalDataReader import CausalDataReader
from faircausal.utils.Dag import find_parents
from faircausal.utils.Metrics import cal_cross_entropy_loss


def negative_log_likelihood(causal_data: CausalDataReader):
    """
    Negative Log-Likelihood for the causal model.

    This function assumes that all continuous variables are normally distributed and all discrete variables are
    dummy-encoded which follow a Binomial or Multinomial distribution.

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
            cross_entropy_loss = cal_cross_entropy_loss(y, y_pred)
            nll_discrete += cross_entropy_loss

    total_nll = nll_continuous + nll_discrete
    return total_nll


def loss(causal_data: CausalDataReader):
    """
    Categorical Cross-Entropy Loss or Mean Squared Error for the causal model.

    :param causal_data: CausalDataReader object
    :return: Error value
    """
    data = causal_data['data']
    beta_dict = causal_data['beta_dict']
    causal_dag = causal_data['causal_dag']
    data_type = causal_data['data_type']
    outcome_variable = causal_data['outcome_variable']

    parents = find_parents(causal_dag, outcome_variable)
    if not parents:
        raise ValueError(f"No parents found for outcome variable {outcome_variable} in causal DAG.")

    X = sm.add_constant(data[parents])
    y_true = data[outcome_variable]

    betas = [beta_dict[f'beta__{outcome_variable}__0']] + [beta_dict[f'beta__{outcome_variable}__{parent}'] for parent
                                                           in parents]

    y_pred = np.dot(X, betas)

    if data_type[outcome_variable] == 'continuous':
        mse_value = mean_squared_error(y_true, y_pred)
        return mse_value
    elif data_type[outcome_variable] == 'discrete':
        cross_entropy_loss = cal_cross_entropy_loss(y_true, y_pred)
        return cross_entropy_loss


def g_formula():
    pass


def nde(causal_data: CausalDataReader):
    pass

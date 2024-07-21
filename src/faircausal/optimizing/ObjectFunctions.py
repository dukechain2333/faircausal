import numpy as np
import statsmodels.api as sm

from faircausal.data.CausalDataReader import CausalDataReader


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
        parents = causal_dag.get(node, [])
        if not parents:
            continue

        X = sm.add_constant(data[parents])
        y = data[node]

        betas = [beta_dict[f'beta__{node}__0']] + [beta_dict[f'beta__{node}__{parent}'] for parent in parents]
        y_pred = np.dot(X, betas)

        if dtype == 'continuous':
            sigma = np.std(y - y_pred)
            # Assuming normal distribution
            nll_continuous += np.sum(0.5 * np.log(2 * np.pi * sigma ** 2) + ((y - y_pred) ** 2) / (2 * sigma ** 2))

        elif dtype == 'discrete':
            n = 1  # Assuming binary outcome
            p = 1 / (1 + np.exp(-y_pred))  # Sigmoid function for logistic regression
            nll_discrete += -np.sum(y * np.log(p) + (n - y) * np.log(1 - p))

    total_nll = nll_continuous + nll_discrete
    return total_nll

from faircausal.optimizing.ObjectFunctions import *
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def eval_f(causal_data: CausalDataReader, lambda_value: float):
    return lambda_value * nde() + negative_log_likelihood(causal_data)


def optimize(causal_data: CausalDataReader):

    lambda_vals = np.arange(0, 1501, 10)
    results = pd.DataFrame({'nde': np.zeros(len(lambda_vals)),
                            'nll': np.zeros(len(lambda_vals)),
                            'mse': np.zeros(len(lambda_vals)),
                            'lambda': lambda_vals})

    for i, lambda_val in enumerate(lambda_vals):
        res = minimize(fun=eval_f,
                       x0=causal_data['beta_dict'],
                       args=(causal_data['data'], lambda_val),
                       method='COBYLA',
                       options={'tol': 1.0e-8,
                                'maxiter': 10000})

        causal_data['beta_dict'] = res.x

        results.at[i, 'nde'] = nde(causal_data)
        results.at[i, 'nll'] = negative_log_likelihood(causal_data)
        results.at[i, 'loss'] = loss(causal_data)

    return results


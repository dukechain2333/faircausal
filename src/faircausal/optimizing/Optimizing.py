from sklearn.linear_model import LogisticRegression, LinearRegression

from faircausal.optimizing.ObjectFunctions import *
import numpy as np
from scipy.optimize import minimize


# def eval_f(causal_data: CausalDataReader, lambda_value: float, exposure: str):
#     return lambda_value * nde(causal_data, exposure) + negative_log_likelihood(causal_data)
#
#
# def optimize(causal_data: CausalDataReader, exposure: str):
#
#     lambda_vals = np.arange(0, 1501, 10)
#     results = pd.DataFrame({'nde': np.zeros(len(lambda_vals)),
#                             'nll': np.zeros(len(lambda_vals)),
#                             'mse': np.zeros(len(lambda_vals)),
#                             'lambda': lambda_vals})
#
#     for i, lambda_val in enumerate(lambda_vals):
#         res = minimize(fun=eval_f,
#                        x0=causal_data['beta_dict'],
#                        args=(causal_data['data'], lambda_val, exposure),
#                        method='COBYLA',
#                        options={'tol': 1.0e-8,
#                                 'maxiter': 10000})
#
#         causal_data['beta_dict'] = res.x
#
#         results.at[i, 'nde'] = nde(causal_data)
#         results.at[i, 'nll'] = negative_log_likelihood(causal_data)
#         results.at[i, 'loss'] = loss(causal_data)
#
#     return results

def initialize_parameters(causal_data: CausalDataReader):
    """
    Initialize the parameter vector and mapping for optimization.

    :param causal_data: CausalDataReader object
    :return: parameter_vector, parameter_mapping
    """
    parameter_vector = []
    parameter_mapping = {}
    index = 0

    for node in causal_data.causal_dag.keys():
        parents = find_parents(causal_data.causal_dag, node)
        num_parents = len(parents)
        num_params = num_parents + 1  # including intercept

        # For mapping, store the slice indices
        parameter_mapping[node] = {
            'start': index,
            'end': index + num_params,
            'parents': parents,
            'type': 'categorical' if causal_data.data[node].dtype.name == 'category' or causal_data.data[node].nunique() == 2 else 'continuous'
        }
        index += num_params

        # Extract initial parameters from the model if available
        if node in causal_data.linear_models:
            model = causal_data.linear_models[node]
            if isinstance(model, LogisticRegression):
                # For LogisticRegression
                intercept = model.intercept_[0] if model.intercept_.shape else model.intercept_
                coef = model.coef_[0]
                params = np.concatenate(([intercept], coef))
            elif isinstance(model, LinearRegression):
                # For LinearRegression
                intercept = model.intercept_
                coef = model.coef_
                params = np.concatenate(([intercept], coef))
            else:
                # Default initialization
                params = np.zeros(num_params)
        else:
            # Default initialization
            params = np.zeros(num_params)

        parameter_vector.extend(params)

    parameter_vector = np.array(parameter_vector)
    return parameter_vector, parameter_mapping


def eval_f(parameter_vector, causal_data: CausalDataReader, parameter_mapping, lambda_value: float, exposure: str):
    """
    Evaluate the penalized objective function.

    :param parameter_vector: Parameter vector
    :param causal_data: CausalDataReader object
    :param parameter_mapping: Mapping of parameters
    :param lambda_value: Penalty parameter
    :param exposure: The name of the exposure variable
    :return: Penalized objective function value
    """
    nll = negative_log_likelihood_param(causal_data, parameter_vector, parameter_mapping)
    nde_value = nde_param(causal_data, parameter_vector, parameter_mapping, exposure)
    penalty = lambda_value * abs(nde_value)
    return nll + penalty

def optimize(causal_data: CausalDataReader, exposure: str, lambda_value: float):
    """
    Optimize the parameters to minimize the penalized objective function.

    :param causal_data: CausalDataReader object
    :param exposure: The name of the exposure variable
    :param lambda_value: Penalty parameter
    :return: Optimized parameter vector and updated causal_data with new parameters
    """
    # Initialize parameters and mapping
    parameter_vector, parameter_mapping = initialize_parameters(causal_data)

    # Define the objective function for optimization
    def objective_func(params):
        return eval_f(params, causal_data, parameter_mapping, lambda_value, exposure)

    # Perform optimization
    result = minimize(objective_func, parameter_vector, method='COBYLA', options={'maxiter': 1000})

    # Update the linear_models in causal_data with optimized parameters
    optimized_params = result.x
    for node in causal_data.causal_dag.keys():
        mapping = parameter_mapping[node]
        start = mapping['start']
        end = mapping['end']
        params = optimized_params[start:end]
        intercept = params[0]
        coefficients = params[1:]
        parents = mapping['parents']

        # Create new models with optimized parameters
        if mapping['type'] == 'categorical':
            # Logistic Regression
            model = LogisticRegression()
            model.intercept_ = np.array([intercept])
            model.coef_ = np.array([coefficients])
        else:
            # Linear Regression
            model = LinearRegression()
            model.intercept_ = intercept
            model.coef_ = coefficients

        # Update the model in causal_data
        causal_data.linear_models[node] = model

    return optimized_params, causal_data



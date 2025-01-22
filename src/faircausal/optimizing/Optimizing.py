from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, LinearRegression

from faircausal.optimizing.ObjectFunctions import negative_log_likelihood_param, nde_param
from faircausal.utils.Dag import find_parents
from faircausal.utils.Evaluation import eval_loss


def initialize_parameters(causal_data):
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
            'type': 'categorical' if causal_data.data[node].dtype.name == 'category' or causal_data.data[
                node].nunique() == 2 else 'continuous'
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


def eval_f(parameter_vector, causal_data, parameter_mapping, lambda_value: float):
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
    nde_value = nde_param(causal_data, parameter_vector, parameter_mapping)
    penalty = lambda_value * abs(nde_value)
    return nll + penalty


def optimize(causal_data, lambda_value: float):
    """
    Optimize the parameters to minimize the penalized objective function.

    :param causal_data: CausalDataReader object
    :param lambda_value: Penalty parameter
    :return: Optimized parameter vector and updated causal_data with new parameters
    """
    parameter_vector, parameter_mapping = initialize_parameters(causal_data)

    def objective_func(params):
        return eval_f(params, causal_data, parameter_mapping, lambda_value)

    result = minimize(objective_func, parameter_vector, method='COBYLA', options={'maxiter': 1000})

    optimized_params = result.x

    optimized_dict = {}

    # Update the linear_models in causal_data with optimized parameters
    for node in causal_data.causal_dag.keys():
        mapping = parameter_mapping[node]
        start = mapping['start']
        end = mapping['end']
        params = optimized_params[start:end]

        intercept = params[0]
        coefficients = params[1:]
        parents = mapping['parents']

        intercept_key = f"beta_{node}_0"
        optimized_dict[intercept_key] = intercept

        for i, parent in enumerate(parents):
            coef_key = f"beta_{node}_{parent}"
            optimized_dict[coef_key] = coefficients[i]

        # Update the scikit-learn model
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

        causal_data.linear_models[node] = model

    return optimized_dict, causal_data


def run_optimize_process(causal_data, lambda_value, run_id):
    optimized_params, causal_data_optimized = optimize(causal_data, lambda_value)

    parameter_vector, parameter_mapping = initialize_parameters(causal_data_optimized)
    nde_param_val = nde_param(causal_data, parameter_vector, parameter_mapping)
    loss = eval_loss(causal_data_optimized)

    return {
        'run_id': run_id,
        'lambda_value': lambda_value,
        'optimized_params': optimized_params,
        'nde_param_val': nde_param_val,
        'loss': loss,
        'causal_data_optimized': causal_data_optimized
    }


def batch_optimize(causal_data, lambda_values: list):
    """
    Batch optimization of the parameters.

    :param causal_data: CausalDataReader object
    :param lambda_values: List of Penalty parameter
    :return: List of optimized parameter dictionaries
    """

    tasks = [(causal_data, lv, i) for i, lv in enumerate(lambda_values)]

    with Pool() as pool:
        batch_optimized_results = pool.starmap(run_optimize_process, tasks)

    return batch_optimized_results

import itertools

import numpy as np


def predict_node(node, data_df, parameter_vector, parameter_mapping):
    """
    Vectorized prediction for a single node over the entire DataFrame.
    (Non-recursive, uses only the immediate parents' values from data_df.)

    :param node: The node (string) to predict.
    :param data_df: Pandas DataFrame containing all rows (columns for node, parents, etc.).
    :param parameter_vector: Numpy array of parameters (flattened).
    :param parameter_mapping: Dict mapping each node to slice indices, parents, and type.
    :return: Numpy array of predictions (one per row).
    """
    # Get metadata for this node
    mapping = parameter_mapping[node]
    start = mapping['start']
    end = mapping['end']
    parents = mapping['parents']
    node_type = mapping['type']

    # Extract intercept & coefficients
    params = parameter_vector[start:end]
    intercept = params[0]
    coefficients = params[1:]

    # Build the matrix
    parent_matrix = data_df[parents].to_numpy(dtype=np.float64)

    # Compute linear output
    linear_output = intercept + np.dot(parent_matrix, coefficients)

    if node_type == 'categorical':
        # Logistic
        y_pred = 1.0 / (1.0 + np.exp(-linear_output))
    else:
        # Linear
        y_pred = linear_output

    return y_pred


def negative_log_likelihood_param(causal_data, parameter_vector, parameter_mapping):
    """
    Negative Log-Likelihood for the causal model using a vectorized approach.

    :param causal_data: CausalDataReader object (assumes .data is a DataFrame)
    :param parameter_vector: Flattened NumPy array of parameters
    :param parameter_mapping: Mapping with slices, parents, type, etc.
    :return: Negative Log-Likelihood (float)
    """
    data_df = causal_data.data
    dag = causal_data.causal_dag

    total_nll = 0.0

    # For each node, predict in batch, then compute contribution to NLL
    for node in dag.keys():
        # Actual values
        actual_vals = data_df[node].to_numpy()

        # Predicted
        y_pred = predict_node(node, data_df, parameter_vector, parameter_mapping)

        # Check node type
        if parameter_mapping[node]['type'] == 'categorical':
            # Binary logistic => compute cross-entropy
            y_pred_clipped = np.clip(y_pred, 1e-10, 1 - 1e-10)  # avoid log(0)
            # actual_vals assumed 0 or 1
            # nll = - sum( y*log(p) + (1-y)*log(1-p) )
            nll_array = -(actual_vals * np.log(y_pred_clipped) + (1.0 - actual_vals) * np.log(1.0 - y_pred_clipped))
        else:
            # Continuous => assume Gaussian with sigma=1
            # nll = sum( 0.5*log(2*pi*sigma^2) + (y-mean)^2 / (2*sigma^2) )
            residuals = actual_vals - y_pred
            nll_array = 0.5 * np.log(2.0 * np.pi * 1.0) + (residuals ** 2) / 2.0

        total_nll += np.sum(nll_array)

    return total_nll


def nde_param(causal_data, parameter_vector, parameter_mapping):
    """
    Calculate the Natural Direct Effect (NDE) with support for both discrete and continuous mediators.

    :param causal_data: CausalDataReader object
    :param parameter_vector: Flattened parameter vector
    :param parameter_mapping: Mapping of parameters
    :return: Estimated NDE
    """

    data_df = causal_data.data.copy()
    outcome_node = causal_data.outcome_variable
    exposure = causal_data.exposure
    mediators = causal_data.mediator

    # Identify continuous and discrete mediators
    discrete_mediators = [m for m in mediators if parameter_mapping[m]['type'] == 'categorical']
    continuous_mediators = [m for m in mediators if parameter_mapping[m]['type'] == 'continuous']

    a = 1
    a_star = 0
    N = len(data_df)

    # Compute P(M_d | A=a*, X) for discrete mediators
    data_astar = data_df.copy()
    data_astar[exposure] = a_star

    p_m_astar = {}
    for mediator in discrete_mediators:
        p_m_astar[mediator] = np.clip(predict_node(mediator, data_astar, parameter_vector, parameter_mapping), 0.0, 1.0)

    # Compute joint probabilities for all combinations of discrete mediators
    mediator_combinations = list(itertools.product([0, 1], repeat=len(discrete_mediators)))

    # Compute P(M_d | A=a*) (joint probability for discrete mediators)
    p_m_comb_astar = np.ones((N, len(mediator_combinations)))
    for idx, combination in enumerate(mediator_combinations):
        for j, mediator in enumerate(discrete_mediators):
            p_m_comb_astar[:, idx] *= np.where(combination[j] == 1, p_m_astar[mediator], 1 - p_m_astar[mediator])

    # Predict E[M_c | A=a*] for continuous mediators
    predicted_m_c_astar = {}
    for mediator in continuous_mediators:
        predicted_m_c_astar[mediator] = predict_node(mediator, data_astar, parameter_vector, parameter_mapping)

    # Predict Y(a=1, M_d, M_c) for all combinations of mediators
    y_a_m_comb = np.zeros((N, len(mediator_combinations)))
    y_astar_m_comb = np.zeros((N, len(mediator_combinations)))

    for idx, combination in enumerate(mediator_combinations):
        data_a_m = data_df.copy()
        data_a_m[exposure] = a
        for j, mediator in enumerate(discrete_mediators):
            data_a_m[mediator] = combination[j]
        for mediator in continuous_mediators:
            data_a_m[mediator] = predicted_m_c_astar[mediator]  # Use predicted mean value

        y_a_m_comb[:, idx] = predict_node(outcome_node, data_a_m, parameter_vector, parameter_mapping)

        data_astar_m = data_df.copy()
        data_astar_m[exposure] = a_star
        for j, mediator in enumerate(discrete_mediators):
            data_astar_m[mediator] = combination[j]
        for mediator in continuous_mediators:
            data_astar_m[mediator] = predicted_m_c_astar[mediator]

        y_astar_m_comb[:, idx] = predict_node(outcome_node, data_astar_m, parameter_vector, parameter_mapping)

    # Compute E[Y(a, M(a*))] and E[Y(a*, M(a*))]
    y_a_m_astar = np.sum(y_a_m_comb * p_m_comb_astar, axis=1)
    y_astar_m_astar = np.sum(y_astar_m_comb * p_m_comb_astar, axis=1)

    # Compute the average NDE
    nde_value = np.mean(y_a_m_astar - y_astar_m_astar)

    return nde_value

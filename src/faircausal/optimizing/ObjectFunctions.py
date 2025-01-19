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
            nll_array = 0.5 * np.log(2.0 * np.pi * 1.0) + (residuals**2) / 2.0

        total_nll += np.sum(nll_array)

    return total_nll


def nde_param(causal_data, parameter_vector, parameter_mapping):
    """
    Calculate the Natural Direct Effect (NDE)

    :param causal_data: CausalDataReader object
    :param parameter_vector: Flattened param vector
    :param parameter_mapping: Mapping of parameters
    :return: Estimated NDE
    """

    data_df = causal_data.data.copy()
    outcome_node = causal_data.outcome_variable
    exposure = causal_data.exposure
    mediator = causal_data.mediator

    a = 1
    a_star = 0
    N = len(data_df)

    # Compute p(M=1|A=a*, X)
    data_astar = data_df.copy()
    data_astar[exposure] = a_star
    # Vectorized probability that M=1
    p_m1_astar = predict_node(mediator, data_astar, parameter_vector, parameter_mapping)
    p_m1_astar = np.clip(p_m1_astar, 0.0, 1.0)  # ensure it's [0,1]
    p_m0_astar = 1.0 - p_m1_astar

    # Predict Y(a=1, M=0) and Y(a=1, M=1)
    data_a_m0 = data_df.copy()
    data_a_m0[exposure] = a
    data_a_m0[mediator] = 0
    y_a_m0 = predict_node(outcome_node, data_a_m0, parameter_vector, parameter_mapping)

    data_a_m1 = data_df.copy()
    data_a_m1[exposure] = a
    data_a_m1[mediator] = 1
    y_a_m1 = predict_node(outcome_node, data_a_m1, parameter_vector, parameter_mapping)

    # Weighted: E[Y(a, M(a*))] = y_a_m0 * p(M=0|a*) + y_a_m1 * p(M=1|a*)
    y_a_m_astar = y_a_m0 * p_m0_astar + y_a_m1 * p_m1_astar

    # Predict Y(a_star=0, M=0) and Y(a_star=0, M=1)
    data_astar_m0 = data_df.copy()
    data_astar_m0[exposure] = a_star
    data_astar_m0[mediator] = 0
    y_astar_m0 = predict_node(outcome_node, data_astar_m0, parameter_vector, parameter_mapping)

    data_astar_m1 = data_df.copy()
    data_astar_m1[exposure] = a_star
    data_astar_m1[mediator] = 1
    y_astar_m1 = predict_node(outcome_node, data_astar_m1, parameter_vector, parameter_mapping)

    # Weighted: E[Y(a*, M(a*))] = y_astar_m0*p(M=0|a*) + y_astar_m1*p(M=1|a*)
    y_astar_m_astar = y_astar_m0 * p_m0_astar + y_astar_m1 * p_m1_astar

    # NDE = average difference
    nde_array = y_a_m_astar - y_astar_m_astar
    nde_value = np.mean(nde_array)

    return nde_value

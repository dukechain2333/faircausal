import numpy as np
from sklearn.metrics import mean_squared_error, log_loss

from faircausal.data.CausalDataReader import CausalDataReader
from faircausal.utils.Dag import recursive_predict, classify_confounders_mediators, find_parents


def negative_log_likelihood(causal_data: CausalDataReader):
    """
    Negative Log-Likelihood for the causal model.

    :param causal_data: CausalDataReader object
    :return: Negative Log-Likelihood
    """
    data = causal_data.data
    linear_models = causal_data.linear_models
    causal_dag = causal_data.causal_dag

    total_nll = 0

    for node in linear_models.keys():

        y = data[node]

        if data[node].dtype.name == 'category':
            y_pred_prob = recursive_predict(node, causal_dag, linear_models, data, final_predict_proba=True)
            total_nll += -np.sum(np.log(y_pred_prob[np.arange(len(y)), y]))
        else:
            y_pred = recursive_predict(node, causal_dag, linear_models, data)
            sigma = np.std(y - y_pred)
            total_nll += np.sum(0.5 * np.log(2 * np.pi * sigma ** 2) + ((y - y_pred) ** 2) / (2 * sigma ** 2))

    return total_nll


def loss(causal_data: CausalDataReader):
    """
    Categorical Cross-Entropy Loss or Mean Squared Error for the causal model.

    :param causal_data: CausalDataReader object
    :return: Loss value
    """
    data = causal_data.data
    linear_models = causal_data.linear_models
    causal_dag = causal_data.causal_dag
    outcome_variable = causal_data.outcome_variable

    if data[outcome_variable].dtype.name == 'category':
        y_pred_prob = recursive_predict(outcome_variable, causal_dag, linear_models, data, final_predict_proba=True)
        return log_loss(data[outcome_variable], y_pred_prob)
    else:
        y_pred = recursive_predict(outcome_variable, causal_dag, linear_models, data)
        return mean_squared_error(data[outcome_variable], y_pred)


def nde(causal_data: CausalDataReader, exposure: str):
    """
    Calculate the Natural Direct Effect (NDE) of the exposure variable on the outcome variable.

    :param causal_data: CausalDataReader object containing the causal DAG, data, and linear models.
    :param exposure: The name of the exposure variable.
    :return: Estimated NDE value.
    """
    data = causal_data.data
    causal_dag = causal_data.causal_dag
    linear_models = causal_data.linear_models
    outcome_variable = causal_data.outcome_variable

    mediator_info = classify_confounders_mediators(causal_dag, exposure, outcome_variable)
    mediators = mediator_info['mediators']

    if not mediators:
        total_effect = (recursive_predict(outcome_variable, causal_dag, linear_models, data, predicted_data={exposure: 1})
                        - recursive_predict(outcome_variable, causal_dag, linear_models, data, predicted_data={exposure: 0}))
        return np.mean(total_effect)

    # M(E=0)
    predicted_mediator_at_exposure_0 = {}
    for mediator in mediators:
        predicted_mediator_at_exposure_0[mediator] = recursive_predict(
            mediator, causal_dag, linear_models, data, final_predict_proba=False
        )

    # Y(E=1, M=M(E=0))
    y_pred_at_exposure_1_mediator_0 = recursive_predict(
        outcome_variable, causal_dag, linear_models, data,
        predicted_data={**predicted_mediator_at_exposure_0, exposure: 1},
        final_predict_proba=False
    )

    # Y(E=0, M=M(E=0))
    y_pred_at_exposure_0_mediator_0 = recursive_predict(
        outcome_variable, causal_dag, linear_models, data,
        predicted_data={**predicted_mediator_at_exposure_0, exposure: 0},
        final_predict_proba=False
    )

    # NDE = E[Y(E=1, M=M(E=0))] - E[Y(E=0, M=M(E=0))]
    nde_value = np.mean(y_pred_at_exposure_1_mediator_0 - y_pred_at_exposure_0_mediator_0)

    return nde_value


def predict_node(node, data_row, parameter_vector, parameter_mapping, predicted_values):
    """
    Predict the value for a single node given the parameter vector.

    :param node: The node to predict.
    :param data_row: A single row from the DataFrame.
    :param parameter_vector: The parameter vector.
    :param parameter_mapping: Mapping from nodes to parameter indices and parents.
    :param predicted_values: Dictionary of already predicted values.
    :return: Predicted value for the node.
    """
    if node in predicted_values:
        return predicted_values[node]

    mapping = parameter_mapping[node]
    start = mapping['start']
    end = mapping['end']
    parents = mapping['parents']
    params = parameter_vector[start:end]

    intercept = params[0]
    coefficients = params[1:]

    # Get parent values; recursively predict if necessary
    parent_values = []
    for parent in parents:
        parent_value = predict_node(parent, data_row, parameter_vector, parameter_mapping, predicted_values)
        parent_values.append(parent_value)
    parent_values = np.array(parent_values, dtype=np.float64)

    linear_output = intercept + np.dot(coefficients, parent_values)

    if mapping['type'] == 'categorical':
        # Logistic regression
        y_pred_prob = 1 / (1 + np.exp(-linear_output))
        # Binary output: return probability or class label
        return y_pred_prob
    else:
        # Linear regression
        return linear_output


def negative_log_likelihood_param(causal_data: CausalDataReader, parameter_vector, parameter_mapping):
    """
    Negative Log-Likelihood for the causal model using the parameter vector.

    :param causal_data: CausalDataReader object
    :param parameter_vector: Parameter vector
    :param parameter_mapping: Mapping of parameters
    :return: Negative Log-Likelihood
    """
    data = causal_data.data
    total_nll = 0

    for index, data_row in data.iterrows():
        predicted_values = {}
        nll = 0

        for node in causal_data.causal_dag.keys():
            actual_value = data_row[node]
            y_pred = predict_node(node, data_row, parameter_vector, parameter_mapping, predicted_values)

            if parameter_mapping[node]['type'] == 'categorical':
                # For categorical variables, compute log-likelihood
                y_true = actual_value
                y_pred_prob = y_pred
                # Avoid numerical issues
                y_pred_prob = np.clip(y_pred_prob, 1e-10, 1 - 1e-10)
                if y_true == 1:
                    nll += -np.log(y_pred_prob)
                else:
                    nll += -np.log(1 - y_pred_prob)
            else:
                # For continuous variables, assume Gaussian distribution
                y_true = actual_value
                y_pred_mean = y_pred
                sigma = 1
                nll += 0.5 * np.log(2 * np.pi * sigma ** 2) + ((y_true - y_pred_mean) ** 2) / (2 * sigma ** 2)

        total_nll += nll

    return total_nll


def nde_param(causal_data: CausalDataReader, parameter_vector, parameter_mapping, exposure: str):
    """
    Calculate the Natural Direct Effect (NDE) using the parameter vector.

    :param causal_data: CausalDataReader object
    :param parameter_vector: Parameter vector
    :param parameter_mapping: Mapping of parameters
    :param exposure: The name of the exposure variable.
    :return: Estimated NDE value.
    """
    data = causal_data.data
    outcome_variable = causal_data.outcome_variable

    mediator_info = classify_confounders_mediators(causal_data.causal_dag, exposure, outcome_variable)
    mediators = mediator_info['mediators']

    nde_values = []

    for index, data_row in data.iterrows():
        predicted_values = {}

        # Predict mediators at exposure=0
        data_row_e0 = data_row.copy()
        data_row_e0[exposure] = 0
        predicted_mediators_e0 = {}
        for mediator in mediators:
            predicted_values_m = {}
            m_pred = predict_node(mediator, data_row_e0, parameter_vector, parameter_mapping, predicted_values_m)
            predicted_mediators_e0[mediator] = m_pred

        # Predict outcome at exposure=1, mediators from exposure=0
        data_row_e1_m0 = data_row.copy()
        data_row_e1_m0[exposure] = 1
        for mediator in mediators:
            data_row_e1_m0[mediator] = predicted_mediators_e0[mediator]
        predicted_values = {}
        y_pred_e1_m0 = predict_node(outcome_variable, data_row_e1_m0, parameter_vector, parameter_mapping, predicted_values)

        # Predict outcome at exposure=0, mediators from exposure=0
        data_row_e0_m0 = data_row.copy()
        data_row_e0_m0[exposure] = 0
        for mediator in mediators:
            data_row_e0_m0[mediator] = predicted_mediators_e0[mediator]
        predicted_values = {}
        y_pred_e0_m0 = predict_node(outcome_variable, data_row_e0_m0, parameter_vector, parameter_mapping, predicted_values)

        nde = y_pred_e1_m0 - y_pred_e0_m0
        nde_values.append(nde)

    return np.mean(nde_values)


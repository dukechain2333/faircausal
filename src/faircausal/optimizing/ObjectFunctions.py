import numpy as np
from sklearn.metrics import mean_squared_error, log_loss

from faircausal.data.CausalDataReader import CausalDataReader
from faircausal.utils.Dag import recursive_predict, classify_confounders_mediators


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


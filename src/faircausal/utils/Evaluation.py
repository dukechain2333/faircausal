from faircausal.data import CausalDataReader
from faircausal.optimizing import initialize_parameters, predict_node

from sklearn.metrics import mean_squared_error, accuracy_score

def prediction(causal_data: CausalDataReader):
    """
    Update the prediction of the causal model.

    :param causal_data: CausalDataReader object

    :return: CausalDataReader object with updated prediction
    """

    parameter_vector, parameter_mapping = initialize_parameters(causal_data)

    outcome = causal_data.outcome_variable

    causal_data.predicted_data = causal_data.data.copy()

    causal_data.predicted_data[outcome] = predict_node(outcome, causal_data.data, parameter_vector, parameter_mapping)

    return causal_data


def eval_loss(causal_data: CausalDataReader):
    """
    Compute the loss of the causal model.

    :param causal_data: CausalDataReader object

    :return: Loss value
    """

    if causal_data.predicted_data is None:
        raise ValueError("Predicted data is not available.")

    outcome = causal_data.outcome_variable

    if causal_data.data[outcome].dtype.name == 'category':
        loss = accuracy_score(causal_data.data[outcome], causal_data.predicted_data[outcome])
    else:
        loss = mean_squared_error(causal_data.data[outcome], causal_data.predicted_data[outcome])

    return loss

from faircausal.data import CausalDataReader
from faircausal.optimizing import initialize_parameters, predict_node


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

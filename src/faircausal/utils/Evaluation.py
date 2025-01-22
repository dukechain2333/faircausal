from sklearn.metrics import mean_squared_error, accuracy_score
import pandas as pd


def prediction(causal_data):
    """
    Update the prediction of the causal model.

    :param causal_data: CausalDataReader object

    :return: CausalDataReader object with updated prediction
    """
    from faircausal.optimizing import initialize_parameters, predict_node

    parameter_vector, parameter_mapping = initialize_parameters(causal_data)

    outcome = causal_data.outcome_variable

    causal_data.predicted_data = causal_data.data.copy()

    causal_data.predicted_data[outcome] = predict_node(outcome, causal_data.data, parameter_vector, parameter_mapping)

    return causal_data


def eval_loss(causal_data):
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


def extract_batch_result(batch_result):
    """
    Extract the result from the batch optimization.

    :param batch_result: List of results from batch optimization

    :return: DataFrame of the results
    """

    result_df = pd.DataFrame(
        {'RunID': [result['run_id'] for result in batch_result],
         'Lambda': [result['lambda_value'] for result in batch_result],
         'Loss': [result['loss'] for result in batch_result],
         'NDE': [result['nde_param_val'] for result in batch_result]})

    result_df = result_df.sort_values(by='Lambda')

    return result_df


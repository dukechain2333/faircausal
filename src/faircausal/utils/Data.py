import pandas as pd


def classify_variables(df: pd.DataFrame):
    """
    Classify variables in a DataFrame as discrete or continuous based on the number of unique values.
    :param df: DataFrame to classify variables.
    :return: Dictionary with variable names as keys and classifications as values.
    """
    classifications = {}
    for column in df.columns:
        unique_values = df[column].nunique()
        total_values = len(df[column])

        # Set thresholds based on the number of total values
        unique_ratio_threshold = 0.02 if total_values > 100 else 0.05
        unique_value_threshold = 20 if total_values > 100 else 10

        # Classify the variable
        if pd.api.types.is_numeric_dtype(df[column]):
            if unique_values < unique_value_threshold or unique_values / total_values < unique_ratio_threshold:
                classifications[column] = 'discrete'
            else:
                classifications[column] = 'continuous'
        else:
            classifications[column] = 'discrete'

    return classifications


def check_if_dummy(df: pd.DataFrame, data_type: dict):
    """
    Check if variables in a DataFrame are one-hot encoded dummy variables (only 0 and 1).
    :param df: DataFrame to check.
    :param data_type: Dictionary with variable names as keys and classifications as values.
    :return: True if all columns are dummy variables, False otherwise.
    """
    for column in df.columns:
        if data_type[column] == 'discrete':
            if len(df[column].unique()) != 2 or set(df[column].unique()) != {0, 1}:
                return False
    return True

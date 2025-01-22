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


def transform_data(df: pd.DataFrame):
    """
    Transform a DataFrame by converting object and category columns to category codes.
    :param df: DataFrame to transform
    :return: Transformed DataFrame
    """
    for column in df.columns:
        if df[column].dtype == 'object' or df[column].dtype.name == 'category':
            df[column] = df[column].astype('category').cat.codes

    return df

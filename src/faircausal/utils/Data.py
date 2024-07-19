import pandas as pd


def classify_variables(df):
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

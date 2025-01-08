import seaborn as sns
import matplotlib.pyplot as plt


def show_outcome_distribution(causal_data, mode = 0, figsize=(10, 7)):
    """
    Visualize the distribution of the outcome variable.

    :param causal_data: CausalDataReader object
    :param mode: 0 for real data, 1 for predicted data
    :param figsize: Size of the figure as a tuple (width, height)
    """
    if mode == 0:
        data = causal_data.data
    else:
        if causal_data.predicted_data is None:
            raise ValueError("Predicted data is not available.")
        data = causal_data.predicted_data

    outcome_variable = causal_data.outcome_variable
    exposure = causal_data.exposure

    plt.figure(figsize=figsize)
    sns.displot(data=data, x=outcome_variable, hue=exposure, kde=True, alpha=0.47)
    plt.title(f"Distribution of {outcome_variable}")
    plt.show()

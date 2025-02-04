import matplotlib.pyplot as plt
import seaborn as sns


def show_outcome_distribution(causal_data, mode=0, figsize=(10, 7)):
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


def loss_nde_plot(result_df, figsize=(10, 7)):
    """
    Visualize the loss and NDE values from the batch optimization.

    :param result_df: DataFrame of the results
    """
    plt.figure(figsize=figsize)
    sns.lineplot(data=result_df, x='NDE', y='Loss')
    plt.title("Loss vs NDE values")
    plt.show()


def lambda_nde_plot(result_df, figsize=(10, 7)):
    """
    Visualize the lambda and NDE values from the batch optimization.

    :param result_df: DataFrame of the results
    """
    plt.figure(figsize=figsize)
    sns.lineplot(data=result_df, x='Lambda', y='NDE')
    plt.title("Lambda vs NDE values")
    plt.show()


def lambda_loss_plot(result_df, figsize=(10, 7)):
    """
    Visualize the lambda and loss values from the batch optimization.

    :param result_df: DataFrame of the results
    """

    plt.figure(figsize=figsize)
    sns.lineplot(data=result_df, x='Lambda', y='Loss')
    plt.title("Lambda vs Loss values")
    plt.show()


def show_results(result_df, figsize=(10, 5)):
    """
    Visualize the loss vs NDE, lambda vs NDE, and lambda vs loss values from the batch optimization.

    :param result_df: DataFrame of the results
    """

    fig, axs = plt.subplots(1, 3, figsize=figsize)

    sns.lineplot(data=result_df, x='NDE', y='Loss', ax=axs[0])
    axs[0].set_title("Loss vs NDE values")

    sns.lineplot(data=result_df, x='Lambda', y='NDE', ax=axs[1])
    axs[1].set_title("Lambda vs NDE values")

    sns.lineplot(data=result_df, x='Lambda', y='Loss', ax=axs[2])
    axs[2].set_title("Lambda vs Loss values")



import numpy as np
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder


def cal_cross_entropy_loss(y, y_pred):
    """
    Calculate the cross-entropy loss between the true labels and the predicted probabilities.

    :param y: True labels
    :param y_pred: Predicted probabilities
    :return: Cross-entropy loss
    """
    encoder = OneHotEncoder()
    y_true_encoded = encoder.fit_transform(y.values.reshape(-1, 1))

    # Ensure y_pred is 2D
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Compute the softmax of the predicted values
    exp_y_pred = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
    y_pred_proba = exp_y_pred / exp_y_pred.sum(axis=1, keepdims=True)

    # Make sure the predicted probabilities have the same number of classes as the true labels
    if y_pred_proba.shape[1] != y_true_encoded.shape[1]:
        missing_cols = y_true_encoded.shape[1] - y_pred_proba.shape[1]
        y_pred_proba = np.hstack((y_pred_proba, np.zeros((y_pred_proba.shape[0], missing_cols))))

    # Compute the cross-entropy loss
    cross_entropy_loss = log_loss(y_true_encoded, y_pred_proba)

    return cross_entropy_loss

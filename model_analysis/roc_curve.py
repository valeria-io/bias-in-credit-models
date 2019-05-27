import pandas as pd
import numpy as np

from sklearn.metrics import roc_curve, auc
from typing import Tuple


def get_roc_auc_data(actuals: pd.Series, predicted_probabilities: pd.Series) -> \
        Tuple[np.array, np.array, np.array, float]:
    """
    Based on actuals and predicted values, it calculates their false positive rate (fpr), the true positive rate (tpr).
    It also returns the corresponding thresholds used as well as the value for the area under the curve.

    :param actuals: series of actual values indicating whether the loan defaulted or not
    :param predicted_probabilities:  series of predicted probabilities of the loan defaulting

    :return: unique series of false and true positive rates with corresponding series of thresholds and value for total
    area under the curve.
    """
    fpr, tpr, thresholds = roc_curve(actuals, predicted_probabilities, pos_label=1)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, thresholds, auc_score


def split_test_set_by_binary_category(test_df: pd.DataFrame, binary_category_name: str, binary_categories: list) ->\
        Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a test dataframe into two, based on the binary category it belongs to (e.g. female/male).

    :param test_df: test dataframe used by XGBoost model with all loan details
    :param binary_category_name: the categorical column used ofr the split
    :param binary_categories: the binary categories used for the split

    :return: two dataframes filtered based on the binary categories
    """
    category_name_zero, category_name_one = binary_categories[0], binary_categories[1]

    test_zero = test_df[test_df[binary_category_name] == category_name_zero]
    test_one = test_df[test_df[binary_category_name] == category_name_one]

    return test_zero, test_one


def split_test_into_x_y(test_df: pd.DataFrame, target_col_name: str):
    """
    Splits a testing set into a predicitive dataframe X and its corresponding target column y.

    :param test_df: complete testing dataframe
    :param target_col_name: target column indicating whether loan will default or not

    :return: predicitive dataframe X and its series for target column y.
    """
    test_y = test_df[target_col_name]
    test_x = test_df.drop(target_col_name, axis=1)

    return test_x, test_y

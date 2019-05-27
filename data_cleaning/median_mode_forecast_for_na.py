import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

from typing import Tuple


def get_train_tests_sets_for_mode_mean(df_: pd.DataFrame, y_col: str) -> \
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divides dataframe into one where null values are present in y_col and another one where there are no null values in
    y_col. For the dataframe without null values in y_col, it creates a dataframe for training and another one for
    testing.

    :param df_: Dataframe with loans details
    :param y_col: name of the column, whose null values we're trying to fill

    :return: training dataframe and testing daframe with no nulls in y_col as well as dataframe with nulls in y_col
    """
    df_ = add_nearest_z(df_)

    df_without_nan_in_y = df_.dropna(subset=[y_col])

    train_df, test_df = train_test_split(df_without_nan_in_y, test_size=0.3)

    df_with_nan_in_y = df_[df_[y_col].isnull()]

    return train_df, test_df, df_with_nan_in_y


def train_with_mode_group(col_to_be_filled: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[str, float]:
    """
    This function goes through different columns in dataframe and each time it groups the dataframe and calculates the
    most frequent value (mode) for the column we're interested in filling. For each iteration, it keeps track of the
    miss-classification rate. It then returns the grouping column that minimises the missclassificaiton on the test set.

    :param col_to_be_filled: column name that we're interested in filling its null values
    :param train_df: training data
    :param test_df: testing data

    :return: the name of the best grouping column and its corresponding miss-classification rate value
    """
    columns_for_grouping = get_columns_for_grouping()

    columns_for_grouping = [col for col in columns_for_grouping if
                            ((col != col_to_be_filled) & (col != col_to_be_filled + '_z'))]

    miss_classifications = []

    for col in columns_for_grouping:
        mode_by_group_df = train_df.groupby(col)[[col_to_be_filled]].agg(
            lambda x: x.value_counts().index[0]).reset_index()
        test_df = test_df.merge(mode_by_group_df, how='left', suffixes=('', '_pred_with_' + col), on=[col])

        test_df_without_nulls = test_df[[col_to_be_filled, col_to_be_filled + '_pred_with_' + col]].dropna()
        accuracy = accuracy_score(test_df_without_nulls[col_to_be_filled],
                                  test_df_without_nulls[col_to_be_filled + '_pred_with_' + col])

        miss_classifications.append(1 - accuracy)

    best_missclassification_on_test = min(miss_classifications)
    optimal_grouping_col = columns_for_grouping[miss_classifications.index(best_missclassification_on_test)]

    return optimal_grouping_col, best_missclassification_on_test


def train_with_median_group(col_to_be_filled: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[str, float]:
    """
    This function goes through different columns in dataframe and each time it groups the dataframe and calculates the
    median for the column we're interested in filling. For each iteration, it keeps track of the mean absolute error
    (MAE). It then returns the grouping column that minimises MAE on the test set.

    :param col_to_be_filled: column name that we're interested in filling its null values
    :param train_df: training data
    :param test_df: testing data

    :return: the name of the best grouping column and its corresponding MAE value
    """
    columns_for_grouping = get_columns_for_grouping()

    columns_for_grouping = [col for col in columns_for_grouping if
                            ((col != col_to_be_filled) & (col != col_to_be_filled + '_z'))]

    mae = []

    for col in columns_for_grouping:
        median_by_group_df = train_df.groupby(col)[[col_to_be_filled]].median()
        test_df = test_df.merge(median_by_group_df, how='left', suffixes=('', '_pred_with_' + col), on=[col])

        test_df_without_nulls = test_df[[col_to_be_filled, col_to_be_filled + '_pred_with_' + col]].dropna()
        mae_score = mean_absolute_error(test_df_without_nulls[col_to_be_filled],
                                        test_df_without_nulls[col_to_be_filled + '_pred_with_' + col])

        mae.append(mae_score)

    best_mae_on_test = min(mae)
    optimal_grouping_col = columns_for_grouping[mae.index(best_mae_on_test)]

    return optimal_grouping_col, best_mae_on_test


def add_nearest_z(df_: pd.DataFrame) -> pd.DataFrame:
    """
    For each numerical column, it creates a new columns that assigns the the nearest z-value.

    :param df_: Dataframe with loans details

    :return: dataframe with new columns indicating nearest z-value (if they're numerical)
    """
    for col in df_.select_dtypes(['number']):
        df_[col + '_z'] = ((df_[col] - df_[col].mean()) / df_[col].std()).round(0).astype(str)
    return df_


def get_columns_for_grouping():
    return ['NewCreditCustomer', 'Gender', 'UseOfLoan', 'Education', 'MaritalStatus', 'EmploymentStatus',
            'EmploymentDurationCurrentEmployer', 'WorkExperience', 'OccupationArea', 'HomeOwnershipType',
            'CreditScoreEeMini', 'Age_z', 'AppliedAmount_z', 'Amount_z', 'Interest_z', 'LoanDuration_z',
            'MonthlyPayment_z', 'NrOfDependants_z', 'IncomeFromPrincipalEmployer_z', 'IncomeFromPension_z',
            'IncomeFromFamilyAllowance_z', 'IncomeFromSocialWelfare_z', 'IncomeFromLeavePay_z',
            'IncomeFromChildSupport_z', 'IncomeOther_z', 'IncomeTotal_z', 'ExistingLiabilities_z',
            'RefinanceLiabilities_z', 'DebtToIncome_z', 'FreeCash_z', 'NoOfPreviousLoansBeforeLoan_z',
            'AmountOfPreviousLoansBeforeLoan_z', 'PreviousRepaymentsBeforeLoan_z',
            'PreviousEarlyRepaymentsBefoleLoan_z', 'PreviousEarlyRepaymentsCountBeforeLoan_z']


def get_median_prediction_score_on_test(y_train: pd.Series, y_test: pd.Series) -> float:
    """
    Calculates accuracy on test based on prediction from simply taking the middle value (median)

    :param y_train: training list of values for the column whose null values we intend to fill
    :param y_test: testing list of values for the column whose null values we intend to fill. Only used to track the
    number of values in test.

    :return: value for mean absolute error on test
    """
    median_predictions = [np.median(y_train)] * len(y_test)
    score_on_test = mean_absolute_error(y_test, median_predictions)
    return score_on_test


def get_mode_prediction_score_on_test(y_train: pd.Series, y_test: pd.Series) -> float:
    """
    Calculates accuracy on test based on prediction from simply taking the most frequent value (mode)

    :param y_train: training list of values for the column whose null values we intend to fill
    :param y_test: testing list of values for the column whose null values we intend to fill. Only used to track the
    number of values in test.

    :return: value for missclassificatio rate on test
    """
    mode_predictions = [pd.Series(y_train.astype(str)).value_counts().index[0]] * len(y_test)
    score_on_test = accuracy_score(y_test, mode_predictions)
    return score_on_test


def predict_with_median(col_without_na: pd.Series, col_with_na: pd.Series) -> list:
    """
    Predicts values to replace NAs by taking the median of the column without NAs.

    :param col_without_na: series from the column whose NAs we're tyring to replace, with no NAs present
    :param col_with_na: series from the column whose NAs we're tyring to replace, which only contain NAs

    :return: list of predicted values
    """
    median_predictions = [np.median(col_without_na)] * len(col_with_na)
    return median_predictions


def predict_with_mode(col_without_na: pd.Series, col_with_na: pd.Series) -> list:
    """
    Predicts values to replace NAs by taking the mode of the column without NAs.

    :param col_without_na: series from the column whose NAs we're tyring to replace, with no NAs present
    :param col_with_na: series from the column whose NAs we're tyring to replace, which only contain NAs

    :return: list of predicted values
    """
    mode_predictions = [pd.Series(col_without_na.astype(str)).value_counts().index[0]] * len(col_with_na)
    return mode_predictions


def predict_with_median_group(df_with_nan_in_y: pd.DataFrame, train_df: pd.DataFrame, optimal_grouping_col: str,
                              col_to_be_filled: str) -> list:
    """
    Predicts values to replace NAs by taking the median from the corresponding group. If there
    is no data to calculate the group's median, we simply use the median from the whole column.

    :param df_with_nan_in_y: dataframe with null values in the column we're tyring to fill
    :param train_df: complete training dataframe with loans details
    :param optimal_grouping_col: the best column to do the grouping to calculate the median
    :param col_to_be_filled: the column whose null values we're intending to replace

    :return: a list of predicted values
    """
    median_by_group_df = train_df.groupby(optimal_grouping_col)[[col_to_be_filled]].median().reset_index()

    df_with_nan_in_y = df_with_nan_in_y.merge(median_by_group_df, how='left', suffixes=('', '_predicted'),
                                              on=[optimal_grouping_col])

    df_with_nan_in_y[col_to_be_filled + '_predicted'].fillna(train_df[col_to_be_filled].median(), inplace=True)

    return df_with_nan_in_y[col_to_be_filled + '_predicted'].tolist()


def predict_with_mode_group(df_with_nan_in_y: pd.DataFrame, train_df: pd.DataFrame, optimal_grouping_col: str,
                            col_to_be_filled: str) -> list:
    """
    Predicts values to replace NAs by calculating the mode (most frequent value) from the corresponding group. If there
    is no data to calculate the group's mode, we simply use the mode from the whole column.

    :param df_with_nan_in_y: dataframe with null values in the column we're tyring to fill
    :param train_df: complete training dataframe with loans details
    :param optimal_grouping_col: the best column to do the grouping to calculate the median
    :param col_to_be_filled: the column whose null values we're intending to replace

    :return: a list of predicted values
    """
    mode_by_group_df = train_df.groupby(optimal_grouping_col)[[col_to_be_filled]].agg(
        lambda x: x.value_counts().index[0]).reset_index()

    df_with_nan_in_y = df_with_nan_in_y.merge(mode_by_group_df, how='left', suffixes=('', '_predicted'),
                                              on=[optimal_grouping_col])

    df_with_nan_in_y[col_to_be_filled + '_predicted'].fillna(train_df[col_to_be_filled].value_counts().index[0],
                                                             inplace=True)

    return df_with_nan_in_y[col_to_be_filled + '_predicted'].tolist()

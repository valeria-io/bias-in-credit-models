import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from statistics import mode
from numbers import Number

from data_cleaning.pre_process import pre_process_raw_data


def raw_df_pre_process_knn_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Filters and processes data for training of knn
    :param df_raw: original data with loans without processing
    :return: processed dataset for training of knn
    """
    training_columns = ['NewCreditCustomer', 'Age', 'Amount', 'Interest', 'LoanDuration', 'Education',
                        'EmploymentDurationCurrentEmployer', 'IncomeFromPrincipalEmployer',
                        'IncomeFromPension', 'IncomeFromFamilyAllowance', 'NrOfDependants',
                        'IncomeFromSocialWelfare', 'IncomeFromLeavePay', 'IncomeFromChildSupport', 'IncomeOther',
                        'ExistingLiabilities', 'RefinanceLiabilities', 'AmountOfPreviousLoansBeforeLoan',
                        'DebtToIncome', 'FreeCash', 'NoOfPreviousLoansBeforeLoan',
                        'PreviousRepaymentsBeforeLoan', 'isLate']

    df_processed = df_raw[training_columns]
    df_processed = df_processed.replace(-1, np.nan)

    return df_processed


def x_convert_to_num_col(df_: pd.DataFrame) -> pd.DataFrame:
    """
    Selects the the columns needed to train Knn and converts column types when needed
    :param df: dataframe with all data
    :return: dataframe with columns needed for knn and in the right column type format
    """
    df_processed = df_.copy()

    df_processed['NewCreditCustomer'] = df_processed['NewCreditCustomer'].replace({'Existing_credit_customer': 0,
                                                                                   'New_credit_Customer': 1})

    df_processed['Education'] = df_processed['Education'].replace({'Primary': 1, 'Basic': 2, 'Vocational': 3,
                                                                   'Secondary': 4, 'Higher': 5})

    df_processed['EmploymentDurationCurrentEmployer'] = df_processed['EmploymentDurationCurrentEmployer'].replace({
        'TrialPeriod': 1, 'Other': 1, 'UpTo1Year': 2, 'UpTo2Years': 3, 'UpTo3Years': 4, 'UpTo4Years': 5,
        'UpTo5Years': 6, 'MoreThan5Years': 7, 'Retiree': 8
    })

    df_processed[['NrOfDependants', 'Education', 'isLate',
                  'NewCreditCustomer', 'CreditScoreEeMini', 'Age', 'NoOfPreviousLoansBeforeLoan',
                  'EmploymentDurationCurrentEmployer']] = \
        df_processed[['NrOfDependants', 'Education', 'isLate',
                      'NewCreditCustomer', 'CreditScoreEeMini', 'Age', 'NoOfPreviousLoansBeforeLoan',
                      'EmploymentDurationCurrentEmployer']].astype(float)

    df_processed = df_processed.fillna(df_processed.median())

    return df_processed


# @todo: check if it is pd.Series in ytrain
def train_k_nn_classifier(x_train: pd.DataFrame, y_train: pd.Series, x_test, y_test):
    """
    Runs k nearest neighbours classifier with cross validation and finds optimal k that reduces MSE
    :param x_train: training set
    :param y_train: testing set
    :return: value for the optimal k, list of neighbours tested and miss-classification scores for each neighbour as a list
    """
    #@todo: update to np.concatenate([np.arange(5, 50, 5), np.arange(50, 100, 10), np.arange(100, 550, 50)])
    neighbors = np.arange(5, 100, 5)

    cv_scores = []

    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())

    miss_classification_list = [1 - x for x in cv_scores]

    optimal_k = neighbors[miss_classification_list.index(min(miss_classification_list))]

    y_pred = predict_with_knn(optimal_k, x_train, y_train, x_test, is_classifier=True)
    knn_score_on_test = get_knn_score_on_test(y_test, y_pred, is_classifier=True)

    return optimal_k, knn_score_on_test


def train_k_nn_regressor(x_train, y_train, x_test, y_test):
    """
    Runs k nearest neighbours with cross validation and finds optimal k that reduces MSE
    :param X_train: training set
    :param y_train: testing set
    :return: value for the optimal k, list of neighbours tested and MSE scores for each neighbour as a list
    """
    #@todo: update to np.concatenate([np.arange(5, 50, 5), np.arange(50, 100, 10), np.arange(100, 550, 50)])
    neighbors = np.arange(5, 100, 5)

    cv_scores = []

    for k in neighbors:
        knn = KNeighborsRegressor(n_neighbors=k)
        scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='neg_mean_absolute_error')
        cv_scores.append(scores.mean())

    mae_list = [(-1) * x for x in cv_scores]

    optimal_k = neighbors[mae_list.index(min(mae_list))]

    y_pred = predict_with_knn(optimal_k, x_train, y_train, x_test, is_classifier=False)
    knn_score_on_test = get_knn_score_on_test(y_test, y_pred, is_classifier=False)

    return optimal_k, knn_score_on_test


def get_knn_score_on_test(y_test, y_pred, is_classifier):
    if is_classifier:
        knn_test_score = accuracy_score(y_test, y_pred)
    else:
        knn_test_score = mean_absolute_error(y_test, y_pred)

    return knn_test_score


def predict_with_knn(optimal_k: int, x_train: pd.DataFrame, y_train, x_test: pd.DataFrame, is_classifier: bool):
    """
    Predicts values using knn and optimal k
    :param optimal_k: k that reduces mae the most
    :param x_train: training set
    :param y_train: training objective variable
    :param x_test: testing features set
    :param y_test: testing objective variable
    :param classifier: indicates wether knn is for Classifier or Regression
    :return: predictions and knn prediction accuracy
    """
    if is_classifier:
        knn = KNeighborsClassifier(n_neighbors=optimal_k)
    else:
        knn = KNeighborsRegressor(n_neighbors=optimal_k)

    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_test)

    return y_pred


def get_train_tests_sets_for_knn(df_, y_col):
    """
    Creates train and test tests for knn. Training set is the df with rows, where there are no null values in y_col.
    Testing set is the df with rows, where there are only null values in y_col. Missing values in X are filled with
    corresponding median.
    :param df: dataframe used to run knn
    :param y_col: the column we're trying to predict
    :return: train and test dataframes for X and y
    """
    df_without_nan_in_y = df_.dropna(subset=[y_col])

    y = df_without_nan_in_y[y_col]
    x = x_convert_to_num_col(df_without_nan_in_y)
    x = raw_df_pre_process_knn_df(x)

    x = x.fillna(x.median())
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    df_with_nan_in_y = df_[df_[y_col].isnull()]
    y_for_pred = df_with_nan_in_y[y_col]

    x_for_pred = x_convert_to_num_col(df_with_nan_in_y)
    x_for_pred = raw_df_pre_process_knn_df(x_for_pred)

    x_for_pred = x_for_pred.fillna(x.median())

    return x_train, x_test, x_for_pred, y_train, y_test, y_for_pred


def get_train_tests_sets_for_mode_mean(df_, y_col):
    df_ = add_nearest_z(df_)

    df_without_nan_in_y = df_.dropna(subset=[y_col])

    train_df, test_df = train_test_split(df_without_nan_in_y, test_size=0.3)

    df_with_nan_in_y = df_[df_[y_col].isnull()]

    return train_df, test_df, df_with_nan_in_y



def fill_na(df_):
    df_filled = df_.copy()

    exclude_columns = ['LoanNumber', 'ListedOnUTC', 'UserName', 'LoanDate', 'MaturityDate_Original',
                       'MaturityDate_Last','DateOfBirth', 'DefaultDate', 'CreditScoreEeMini', 'isLate', 'Defaulted',
                       'DefaultStatus', 'AgeGroup']

    columns_to_be_filled = [col for col in df_.columns if (col not in exclude_columns) & (df_[col].isnull().values.any())]

    models_regression = ['knn', 'group_median', 'median']
    models_classifier = ['knn', 'group_mode', 'mode']

    for col in columns_to_be_filled:

        x_train, x_test, x_for_pred, y_train, y_test, y_for_pred = get_train_tests_sets_for_knn(df_, col)
        train_df, test_df, df_with_nan_in_y = get_train_tests_sets_for_mode_mean(df_, col)

        if np.issubdtype(df[col].dtype, np.number):

            optimal_k, best_mae_on_test_knn = train_k_nn_regressor(x_train, y_train, x_test, y_test)
            optimal_grouping_col, best_mae_on_test_group_median = train_with_median_group(col, train_df, test_df)
            best_mae_on_test_median = get_median_prediction_score_on_test(y_train, y_test)

            scores = [best_mae_on_test_knn, best_mae_on_test_group_median, best_mae_on_test_median]
            best_score = min(scores)
            best_model = models_regression[scores.index(best_score)]

            print('[{}] optimal k: {} | optimal group col: {}'.format(col, optimal_k, optimal_grouping_col))
            print('Knn MAE: {} |  Median MAE: {} | Median Group MAE: {}'.format(best_mae_on_test_knn, best_mae_on_test_median,
                                                                                best_mae_on_test_group_median))
            print('Best model: {} | MAE: {}'.format(best_model, best_score))

            y_for_pred = predict_regression(df_with_nan_in_y, col, x_train, y_train, train_df, x_for_pred, best_model, optimal_k,
                               optimal_grouping_col)

        else:
            optimal_k, best_miss_claf_on_test_knn = train_k_nn_classifier(x_train, y_train, x_test, y_test)
            optimal_grouping_col, best_miss_claf_on_test_group_mode = train_with_mode_group(col, train_df, test_df)
            best_miss_claf_on_test_mode = get_mode_prediction_score_on_test(y_train, y_test)

            scores = [best_miss_claf_on_test_knn, best_miss_claf_on_test_group_mode, best_miss_claf_on_test_mode]
            best_score = min(scores)
            best_model = models_classifier[scores.index(best_score)]


            print('[{}] optimal k: {} | optimal group col: {}'.format(col, optimal_k, optimal_grouping_col))
            print('Knn MissClass: {} |  Mode MissClass: {} | Mode Group MissClass: {}'.format(best_miss_claf_on_test_knn,
                                                                                              best_miss_claf_on_test_mode,
                                                                                              best_miss_claf_on_test_group_mode))
            print('Best model: {} | MAE: {}'.format(best_model, best_score))

            y_for_pred = predict_classification(df_with_nan_in_y, col, x_train, y_train, train_df, x_for_pred, best_model, optimal_k,
                                            optimal_grouping_col)

        df_filled.loc[df_filled[col].isnull(), [col]] = y_for_pred

    return df_filled


def add_nearest_z(df_):
    for col in df_.select_dtypes(['number']):
        df_[col + '_z'] = ((df[col] - df[col].mean()) / df[col].std()).round(0).astype(str)
    return df_


def get_columns_for_grouping():
    return ['NewCreditCustomer', 'Gender', 'UseOfLoan', 'Education', 'MaritalStatus', 'EmploymentStatus',
            'EmploymentDurationCurrentEmployer', 'WorkExperience', 'OccupationArea', 'HomeOwnershipType',
            'CreditScoreEeMini', 'Age_z', 'AppliedAmount_z','Amount_z', 'Interest_z', 'LoanDuration_z',
            'MonthlyPayment_z', 'NrOfDependants_z', 'IncomeFromPrincipalEmployer_z', 'IncomeFromPension_z',
            'IncomeFromFamilyAllowance_z', 'IncomeFromSocialWelfare_z', 'IncomeFromLeavePay_z',
            'IncomeFromChildSupport_z', 'IncomeOther_z', 'IncomeTotal_z', 'ExistingLiabilities_z',
            'RefinanceLiabilities_z', 'DebtToIncome_z', 'FreeCash_z', 'NoOfPreviousLoansBeforeLoan_z',
            'AmountOfPreviousLoansBeforeLoan_z', 'PreviousRepaymentsBeforeLoan_z',
            'PreviousEarlyRepaymentsBefoleLoan_z', 'PreviousEarlyRepaymentsCountBeforeLoan_z']


def train_with_mode_group(col_to_be_filled, train_df, test_df):

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


def train_with_median_group(col_to_be_filled, train_df, test_df):

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


def get_median_prediction_score_on_test(y_train, y_test):
    median_predictions = [np.median(y_train)] * len(y_test)
    score_on_test = mean_absolute_error(y_test, median_predictions)
    return score_on_test


def get_mode_prediction_score_on_test(y_train, y_test):
    mode_predictions = [pd.Series(y_train.astype(str)).value_counts().index[0]] * len(y_test)
    score_on_test = accuracy_score(y_test, mode_predictions)
    return score_on_test


def predict_with_median(col_without_na, col_with_na):
    median_predictions = [np.median(col_without_na)] * len(col_with_na)
    return median_predictions


def predict_with_mode(col_without_na, col_with_na):
    mode_predictions = [pd.Series(col_without_na.astype(str)).value_counts().index[0]] * len(col_with_na)
    return mode_predictions


def predict_with_median_group(df_with_nan_in_y, train_df, optimal_grouping_col, col_to_be_filled):
    median_by_group_df = train_df.groupby(optimal_grouping_col)[[col_to_be_filled]].median().reset_index()

    df_with_nan_in_y = df_with_nan_in_y.merge(median_by_group_df, how='left', suffixes=('', '_predicted'),
                                              on=[optimal_grouping_col])

    # If no data avaiable on optimal_grouping_col, just fill with median
    df_with_nan_in_y[col_to_be_filled + '_predicted'].fillna(train_df[col_to_be_filled].median(), inplace=True)

    return df_with_nan_in_y[col_to_be_filled + '_predicted'].tolist()


def predict_with_mode_group(df_with_nan_in_y, train_df, optimal_grouping_col, col_to_be_filled):
    mode_by_group_df = train_df.groupby(optimal_grouping_col)[[col_to_be_filled]].agg(
        lambda x: x.value_counts().index[0]).reset_index()

    df_with_nan_in_y = df_with_nan_in_y.merge(mode_by_group_df, how='left', suffixes=('', '_predicted'),
                                              on=[optimal_grouping_col])

    # If no data avaiable on optimal_grouping_col, just fill with mode
    df_with_nan_in_y[col_to_be_filled + '_predicted'].fillna(train_df[col_to_be_filled].value_counts().index[0], inplace=True)

    return df_with_nan_in_y[col_to_be_filled + '_predicted'].tolist()


def predict_regression(df_with_nan_in_y, col_to_be_filled, x_train, y_train, train_df, x_for_pred, best_model, optimal_k, optimal_grouping_col):
    if best_model == 'knn':
        y_for_pred = predict_with_knn(optimal_k, x_train, y_train, x_for_pred, is_classifier=False)

    elif best_model == 'group_median':
        y_for_pred = predict_with_median_group(df_with_nan_in_y, train_df, optimal_grouping_col, col_to_be_filled)

    else:
        y_for_pred = predict_with_median(x_train, df_with_nan_in_y[col_to_be_filled])
    return y_for_pred


def predict_classification(df_with_nan_in_y, col_to_be_filled, x_train, y_train, train_df, x_for_pred, best_model, optimal_k, optimal_grouping_col):
    if best_model == 'knn':
        y_for_pred = predict_with_knn(optimal_k, x_train, y_train, x_for_pred, is_classifier=True)

    elif best_model == 'group_mode':
        y_for_pred = predict_with_mode_group(df_with_nan_in_y, train_df, optimal_grouping_col, col_to_be_filled)

    else:
        y_for_pred = predict_with_mode(y_train, df_with_nan_in_y[col_to_be_filled])
    return y_for_pred


if __name__ == "__main__":
    df = pd.read_csv("../data/df_selection.csv", index_col=[0])
    df = pre_process_raw_data(df)
    # df_na_filled_with_knn = fill_na_with_k_nearest_neighbours(df, show_k_results=True)
    fill_na(df)
    # df_na_filled.to_csv("../data/df_cleaned.csv")

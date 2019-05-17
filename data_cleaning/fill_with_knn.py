import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error


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
def train_k_nn_classifier(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series):
    """
    Runs k nearest neighbours classifier with cross validation and finds optimal k that reduces MSE
    :param x_train: training set
    :param y_train: testing set
    :return: value for the optimal k, list of neighbours tested and miss-classification scores for each neighbour as a list
    """
    # @todo: update to np.concatenate([np.arange(5, 50, 5), np.arange(50, 100, 10), np.arange(100, 550, 50)])
    neighbors = [5]

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
    # @todo: update to np.concatenate([np.arange(5, 50, 5), np.arange(50, 100, 10), np.arange(100, 550, 50)])
    neighbors = [5]

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




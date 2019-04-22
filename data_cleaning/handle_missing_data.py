import numpy as np

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


def pre_process_knn_df(df):
    """
    Selects the the columns needed to train Knn and converts column types when needed
    :param df: dataframe with all data
    :return: dataframe with columns needed for knn and in the right column type format
    """
    training_columns = ['NewCreditCustomer', 'Age', 'Gender', 'Amount', 'Interest', 'LoanDuration', 'Education',
                       'NrOfDependants', 'EmploymentStatus', 'WorkExperience', 'OccupationArea', 'HomeOwnershipType',
                       'IncomeFromPrincipalEmployer', 'IncomeFromPension', 'IncomeFromFamilyAllowance',
                       'IncomeFromSocialWelfare', 'IncomeFromLeavePay', 'IncomeFromChildSupport', 'IncomeOther',
                       'IncomeTotal', 'ExistingLiabilities', 'RefinanceLiabilities', 'AmountOfPreviousLoansBeforeLoan',
                       'DebtToIncome', 'FreeCash', 'CreditScoreEeMini', 'NoOfPreviousLoansBeforeLoan',
                       'PreviousRepaymentsBeforeLoan', 'isLate']

    df_processed = df[training_columns]

    for col in df_processed.select_dtypes(include=[object]).columns:
        df_processed[col] = df_processed[col].astype('category')
        df_processed[col] = df_processed[col].cat.codes

    df_processed = df_processed.replace(-1, np.nan)

    return df_processed


def train_k_nn(X_train, y_train):
    """
    Runs k nearest neighbours with cross validation and finds optimal k that reduces MSE
    :param X_train: training set
    :param y_train: testing set
    :return: value for the optimal k, list of neighbours tested and MSE scores for each neighbour as a list
    """

    neighbors = np.arange(100,1000,100)

    cv_scores = []

    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())

    mse = [1 - x for x in cv_scores]

    optimal_k = neighbors[mse.index(min(mse))]

    return optimal_k, neighbors, mse


def predict_with_knn(optimal_k, x_train, y_train, x_test, y_test):
    """
    Predicts values using knn and optimal k
    :param optimal_k: k that reduces mse the most
    :param x_train: training features set
    :param y_train: training objective variable
    :param x_test: testing features set
    :param y_test: testing objective variable
    :return: predictions and knn prediction accuracy
    """
    knn = KNeighborsClassifier(n_neighbors=optimal_k)
    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_test)
    knn_test_accuracy_score = accuracy_score(y_test, y_pred)

    return y_pred, knn_test_accuracy_score



def plot_k_nn_results(optimal_k, neighbours, mse, knn_test_accuracy_score):
    """
    Plots MSE for each k and names the optimal k
    :param optimal_k: optimal k value with smallest MSE
    :param neighbours: all neighbours tested
    :param MSE: MSE scores for each neighbour as a list
    :return: plots a graph, prints optimal k and accuracy on test set
    """
    print("The optimal number of neighbors is {}".format(optimal_k))
    plt.plot(neighbours, mse)
    plt.xlabel('Number of Neighbors K')
    plt.ylabel('Misclassification Error')
    plt.show()
    print("Accuracy on test is: {}".format(knn_test_accuracy_score))


def get_train_tests_sets_for_knn(df, y_col):
    """
    Creates train and test tests for knn. Training set is the df with rows, where there are no null values in y_col.
    Testing set is the df with rows, where there are only null values in y_col. Missing values in X are filled with
    corresponding mean.
    :param df: dataframe used to run knn
    :param y_col: the column we're trying to predict
    :return: train and test dataframes for X and y
    """
    df_without_nan_in_y = df.dropna(subset=[y_col])

    x_train = df_without_nan_in_y.drop([y_col], axis=1)
    x_train = x_train.fillna(x_train.mean())

    y_train = df_without_nan_in_y[y_col]

    df_with_nan_in_y = df[df[y_col].isnull()]

    x_test = df_with_nan_in_y.drop([y_col], axis=1)
    x_test = x_test.fillna(x_test.mean())

    y_test = df_with_nan_in_y[y_col]

    return x_train, x_test, y_train, y_test



def fill_na_with_k_nearest_neighbours(df, show_k_results=False):
    df_processed = pre_process_knn_df(df)

    # @todo: check if there is a more elegant way to select columns that contain null values
    columns_with_null_values = [col for col in df_processed.columns if df[col].isnull().any()]


    for col in columns_with_null_values:

        x_train, x_test, y_train, y_test = get_train_tests_sets_for_knn(df_processed, col)

        optimal_k, neighbors, mse = train_k_nn(x_train, y_train)

        y_pred, knn_test_accuracy_score = predict_with_knn(optimal_k, x_train, y_train, x_test, y_test)

        # @todo: fill df with predicted values

        if show_k_results:
            plot_k_nn_results(optimal_k, neighbors, mse, knn_test_accuracy_score)

    return -1

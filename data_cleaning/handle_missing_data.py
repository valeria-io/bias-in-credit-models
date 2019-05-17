import pandas as pd
import numpy as np

from data_cleaning.fill_with_median_mode import \
    get_train_tests_sets_for_mode_mean, \
    get_mode_prediction_score_on_test, \
    get_median_prediction_score_on_test, \
    train_with_median_group, \
    train_with_mode_group, \
    predict_with_mode, \
    predict_with_median, \
    predict_with_median_group, \
    predict_with_mode_group

from data_cleaning.fill_with_knn import \
    get_train_tests_sets_for_knn, \
    train_k_nn_regressor, \
    train_k_nn_classifier,\
    predict_with_knn

from data_cleaning.pre_process import pre_process_raw_data



def fill_na(df_: pd.DataFrame) -> pd.DataFrame:
    """
    Fills null values based in dataframe with loans. To do this, it tries out three different forms of fill the values:
    1) Using K nearest neighbours: regression for numerical columns and classification for non-numerical columns
    2) Taking the most frequent value based on a group: median based on group for numerical columns and mode
    based on group for non-numerical columns
    3) Taking the most frequent value for whole column: median for numerical columns and mode for non-numerical columns

    :param df_: raw dataframe with loans
    :return: dataframe with loans whose null values has been replaced/filled
    """
    df_filled = df_.copy()

    exclude_columns = ['LoanNumber', 'ListedOnUTC', 'UserName', 'LoanDate', 'MaturityDate_Original',
                       'MaturityDate_Last', 'DateOfBirth', 'DefaultDate', 'CreditScoreEeMini', 'isLate', 'Defaulted',
                       'DefaultStatus', 'AgeGroup']

    columns_to_be_filled = [col for col in df_.columns if
                            (col not in exclude_columns) & (df_[col].isnull().values.any())]

    models_regression = ['knn', 'group_median', 'median']
    models_classifier = ['knn', 'group_mode', 'mode']

    for col in columns_to_be_filled[0:15]:

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
            print('Knn MAE: {} |  Median MAE: {} | Median Group MAE: {}'.format(best_mae_on_test_knn,
                                                                                best_mae_on_test_median,
                                                                                best_mae_on_test_group_median))
            print('Best model: {} | MAE: {}'.format(best_model, best_score))

            y_for_pred = predict_regression(best_model, col, x_train, y_train, x_for_pred, optimal_k, train_df,
                                            df_with_nan_in_y, optimal_grouping_col)

        else:
            optimal_k, best_miss_claf_on_test_knn = train_k_nn_classifier(x_train, y_train, x_test, y_test)
            optimal_grouping_col, best_miss_claf_on_test_group_mode = train_with_mode_group(col, train_df, test_df)
            best_miss_claf_on_test_mode = get_mode_prediction_score_on_test(y_train, y_test)

            scores = [best_miss_claf_on_test_knn, best_miss_claf_on_test_group_mode, best_miss_claf_on_test_mode]
            best_score = min(scores)
            best_model = models_classifier[scores.index(best_score)]

            print('[{}] optimal k: {} | optimal group col: {}'.format(col, optimal_k, optimal_grouping_col))
            print('Knn MissClass: {} |  Mode MissClass: {} | Mode Group MissClass: {}'.
                  format(best_miss_claf_on_test_knn, best_miss_claf_on_test_mode, best_miss_claf_on_test_group_mode))
            print('Best model: {} | MAE: {}'.format(best_model, best_score))

            y_for_pred = predict_classification(best_model, col, x_train, y_train, x_for_pred, optimal_k, train_df,
                                                df_with_nan_in_y, optimal_grouping_col)

        df_filled.loc[df_filled[col].isnull(), [col]] = y_for_pred

    return df_filled


def predict_regression(best_model: str, col_to_be_filled: str, x_train: pd.DataFrame, y_train: pd.Series,
                       x_for_pred: pd.DataFrame, optimal_k: int, train_df: pd.DataFrame, df_with_nan_in_y: pd.DataFrame,
                       optimal_grouping_col: str)->list:
    """
    Predicts values for the numerical column whose NAs need to be replaced using the best model to fill null values.

    :param best_model: the best model to fill null values
    :param col_to_be_filled: name of the column that needs to be filled
    :param x_train: dataframe used for training knn
    :param y_train: series of the column of interest for training
    :param x_for_pred: dataframe that will be used by knn for predicting the new values (does not contain col to be filled)
    :param optimal_k: optimal number of neighbours for knn
    :param train_df: complete training dataframe
    :param df_with_nan_in_y: dataframe that contains null values in the column whose null values will be filled
    :param optimal_grouping_col: the optimal column to group and calculate median as a forecast to fill null values
    :return: predicted values used to replace NAs based on the best model fill null values
    """
    if best_model == 'knn':
        y_for_pred = predict_with_knn(optimal_k, x_train, y_train, x_for_pred, is_classifier=False)

    elif best_model == 'group_median':
        y_for_pred = predict_with_median_group(df_with_nan_in_y, train_df, optimal_grouping_col, col_to_be_filled)

    else:
        y_for_pred = predict_with_median(x_train, df_with_nan_in_y[col_to_be_filled])
    return y_for_pred


def predict_classification(best_model: str, col_to_be_filled: str, x_train: pd.DataFrame, y_train: pd.Series,
                           x_for_pred: pd.DataFrame, optimal_k: int, train_df: pd.DataFrame,
                           df_with_nan_in_y: pd.DataFrame, optimal_grouping_col: str)->list:
    """
    Predicts values for the numerical column whose NAs need to be replaced using the best model to fill null values.

    :param best_model: the best model to fill null values
    :param col_to_be_filled: name of the column that needs to be filled
    :param x_train: dataframe used for training knn
    :param y_train: series of the column of interest for training
    :param x_for_pred: dataframe that will be used by knn for predicting the new values (does not contain col to be filled)
    :param optimal_k: optimal number of neighbours for knn
    :param train_df: complete training dataframe
    :param df_with_nan_in_y: dataframe that contains null values in the column whose null values will be filled
    :param optimal_grouping_col: the optimal column to group and calculate median as a forecast to fill null values
    :return: predicted values used to replace NAs based on the best model fill null values
    """
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

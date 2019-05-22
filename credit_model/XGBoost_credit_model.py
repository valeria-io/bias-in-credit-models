import pandas as pd

from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

from credit_model.prepare_model_dataframe import prepare_dataframe


def find_best_xgboost_model(train_x: pd.DataFrame, train_y: pd.Series) -> (dict, float):
    """
    Runs a grid search to find the tuning parameters that maxisimise the area under the curve (AUC)
    :param train_x: training data frame with loan details
    :param train_y: default target column for training
    :return: best parameters and corresponding AUC score
    """
    scale_pos_weight = (len(train_y) - train_y.sum()) / train_y.sum()

    param_test = {
        'max_depth': [8, 9, 10, 11],
        'learning_rate': [0.015, 0.05, 0.1, 0.15],
        'n_estimators': [100, 150, 200, 300, 400, 500]
    }

    gsearch = GridSearchCV(estimator=XGBClassifier(
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=scale_pos_weight,
        seed=27),
        param_grid=param_test, scoring='roc_auc', n_jobs=-1, iid=False, cv=5)

    gsearch.fit(train_x, train_y)

    return gsearch.best_params_, gsearch.best_score_


def xgboost_predict(best_params_: dict, train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame,
                    test_y: pd.Series) -> (list, float):
    """
    Using the xgboost model parameters, it predicts the probabilities of defaulting.
    :param best_params_: best tuning parameters
    :param train_x: training dataframe with loan details
    :param train_y: default target column for training
    :param test_x: testing dataframe with loan details
    :param test_y: default target column for testing
    :return: series of probabilities whether loan entry will default or not and corresponding model's AUC score
    """
    scale_pos_weight = (len(train_y) - train_y.sum()) / train_y.sum()
    xgb_model = XGBClassifier(objective='binary:logistic',
                              scale_pos_weight=scale_pos_weight,
                              seed=27,
                              max_depth=best_params_['max_depth'],
                              learning_rate=best_params_['learning_rate'],
                              n_estimators=best_params_['n_estimators']
                              )

    xgb_model.fit(train_x, train_y)
    predicted_probabilities_ = xgb_model.predict_proba(test_x)[:, 1]
    auc_ = roc_auc_score(test_y, predicted_probabilities_)

    return predicted_probabilities_, auc_


def get_train_test_dataframes(df_: pd.DataFrame, target_variable: str) -> (
pd.DataFrame, pd.DataFrame, pd.Series, pd.Series):
    """
    Splits dataframe into features and target variable for train and test.
    :param df_: dataframe loan details
    :param target_variable: name of default target variable
    :return: x dataframes and y series for train and test
    """
    X = df_.drop(target_variable, axis=1)
    y = df_[target_variable]
    train_x, test_x, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25, random_state=42)
    return train_x, test_x, train_y, test_y


if __name__ == "__main__":
    df = pd.read_csv("../data/LoanDataProcessed.csv")
    df = prepare_dataframe(df)
    train_X, test_X, train_Y, test_Y = get_train_test_dataframes(df, "Defaulted")
    best_params, best_score = find_best_xgboost_model(train_X, train_Y)
    print('Best Parameters: {} | Best AUC: {}'.format(best_params, best_score))
    predicted_probabilities, auc = xgboost_predict(best_params, train_X, train_Y, test_X, test_Y)
    print("AUC: {}".format(auc))
    """
    This returned:
    Best Parameters: {'learning_rate': 0.05, 'max_depth': 9, 'n_estimators': 300} | Best AUC: 0.7570655614472669
    AUC: 0.7638599401900094
    """
    pd.DataFrame({'actuals': test_Y, 'predicted_probabilities': predicted_probabilities}).to_csv(
        '../data/actuals_and_forecasts.csv')

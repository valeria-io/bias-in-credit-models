import pandas as pd

from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from credit_model.prepare_model_dataframe import prepare_dataframe


def find_best_xgboost_model(train_x, train_y):
    scale_pos_weight = (len(train_y) - train_y.sum()) / train_y.sum()

    param_test = {
        'max_depth': range(3, 11),
        'gamma': [i / 10.0 for i in range(0, 5)],
        'min_child_weight': range(1, 15, 1),
        'subsample': [i / 10.0 for i in range(6, 10)],
        'colsample_bytree': [i / 10.0 for i in range(6, 10)],
        'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
    }

    gsearch = GridSearchCV(estimator=XGBClassifier(
        learning_rate=0.01,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=scale_pos_weight,
        seed=27),
        param_grid=param_test, scoring='roc_auc', n_jobs=4, iid=False, cv=10)

    gsearch.fit(train_x, train_y)

    return gsearch.best_params_, gsearch.best_score_


def xgboost_predict(best_params_, train_x, train_y, test_x, test_y):
    scale_pos_weight = (len(train_y) - train_y.sum()) / train_y.sum()
    xgb_model = XGBClassifier(objective='binary:logistic',
                              learning_rate=0.01,
                              n_jobs=4,
                              scale_pos_weight=scale_pos_weight,
                              seed=27,
                              max_depth=best_params_['max_depth'],
                              gamma=best_params_['gamma'],
                              min_child_weight=best_params_['min_child_weight'],
                              subsample=best_params_['subsample'],
                              colsample_bytree=best_params_['colsample_bytree'],
                              reg_alpha=best_params_['reg_alpha'])

    xgb_model.fit(train_x, train_y)

    predictions_ = xgb_model.predict(test_x)
    accuracy_ = accuracy_score(test_y, predictions)

    return predictions_, accuracy_


def get_train_test_dataframes(df_, target_variable):
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
    predictions, accuracy = xgboost_predict(best_params, train_X, train_Y, test_X, test_Y)
    print("Accuracy: {}.2f%%".format(accuracy * 100.0))
    pd.DataFrame({'actuals': test_Y, 'predictions': predictions}).to_csv('../data/actuals_and_forecasts.csv')

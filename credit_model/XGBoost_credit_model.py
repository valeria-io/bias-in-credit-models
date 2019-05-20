import pandas as pd

from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

from credit_model.prepare_model_dataframe import prepare_dataframe


def find_best_xgboost_model(train_x, train_y):
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


def xgboost_predict(best_params_, train_x, train_y, test_x, test_y):
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
    predicted_probabilities, auc = xgboost_predict(best_params, train_X, train_Y, test_X, test_Y)
    print("AUC: {}".format(auc))
    pd.DataFrame({'actuals': test_Y, 'predicted_probabilities': predicted_probabilities}).to_csv('../data/actuals_and_forecasts.csv')

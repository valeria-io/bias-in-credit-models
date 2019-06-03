import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

from model_analysis.roc_curve import split_test_set_by_binary_category

from typing import Tuple


def calculate_classification_metrics(actuals: pd.Series, predicted_proba: pd.Series, threshold: float) -> \
        Tuple[int, int, float, float, float, float]:
    """
    Calculates the metrics for a confusion matrix and the values for the true positive rate and false positive rate
    :param actuals: actual values for the target variable 'Defaulted'
    :param predicted_proba: predicted probability for the target variable
    :param threshold: the threshold used to classify the probability as defaulted or not
    :return: values for (1) the number of true positives (predicted to pay back and pays back), (2) the number of false
    positives (predicted to pay back, but defaults) as well as the aggregate values for (3) the positive rate, (4) the
    negative rate, (5) the true positive rate and (6) the false positive rate
    """

    predicted = predicted_proba.apply(lambda x: 1 if x >= threshold else 0)

    tn, fp, fn, tp = confusion_matrix(actuals, predicted).ravel()

    """ Positive rate: % classified as positive (% predicted to pay back a loan) """
    pr = (tp + fp) / (tn + fp + fn + tp)

    """ Negative rate: % classified as negative (% predicted to default) """
    nr = (tn + fn) / (tn + fp + fn + tp)

    """ True positive rate: % of all positive that were classified correctly to pay back a loan """
    tpr = tp / (tp + fn)

    """ False positive rate: % of all negatives that we miss-classified as being able to pay back a loan """
    fpr = fp / (fp + tn)

    return tp, fp, pr, nr, tpr, fpr


def run_algorithmic_interventions_df(df_dict, col_names_dict, weights_dict):
    results_columns = ['IntervationName', 'Profit', 'threshold_0', 'threshold_1', 'TruePositive0', 'FalsePositive0',
                       'PositiveRate0', 'NegativeRate0', 'TruePositiveRate0', 'FalsePositiveRate0', 'TruePositive1',
                       'FalsePositive1', 'PositiveRate1', 'NegativeRate1', 'TruePositiveRate1', 'FalsePositiveRate1']

    results_df = pd.DataFrame(columns=results_columns)

    thresholds = np.arange(0, 1.01, 0.01)

    df_0 = df_dict['group_0']
    df_1 = df_dict['group_1']

    for t0 in thresholds:
        for t1 in thresholds:

            results_group_0 = calculate_classification_metrics(df_0[col_names_dict['actuals_col_name']],
                                                               df_0[col_names_dict['predicted_col_name']], t0)

            results_group_1 = calculate_classification_metrics(df_1[col_names_dict['actuals_col_name']],
                                                               df_1[col_names_dict['predicted_col_name']], t1)

            tp0, fp0, pr0, nr0, tpr0, fpr0 = results_group_0
            tp1, fp1, pr1, nr1, tpr1, fpr1 = results_group_1

            profit_function = weights_dict['weight_tp'] * (tp0 + tp1) - weights_dict['weight_fp'] * (fp0 + fp1)

            """
            Intervention 1: Maximise profit - uses different or equal thresholds for each category without any
            constrains
            """
            results_df = results_df.append(
                pd.DataFrame(
                    columns=results_columns,
                    data=[('MaxProfit', profit_function, t0, t1) + results_group_0 + results_group_1]))

            """
            Intervention 2: Group unawareness - uses equal threshold for both categories without any constrains
            """
            if t0 == t1:
                results_df = results_df.append(pd.DataFrame(
                    columns=results_columns,
                    data=[('GroupUnawareness', profit_function, t0, t1) + results_group_0 + results_group_1]))

            """
            Intervention 3: Demographic parity - uses different or equal threshold for each category as soon as each
            group gets granted the same percentage of loans (equal positive rate)
            """
            if round(pr0, 2) == round(pr1, 2):
                results_df = results_df.append(pd.DataFrame(
                    columns=results_columns,
                    data=[('DemographicParity', profit_function, t0, t1) + results_group_0 + results_group_1]))

            """
            Intervention 4: Equal Opportunity - uses different or equal thresholds for each category as soon as each
            group has the same rate of correctly classified loans as paid (equal TPR)
            """
            if round(tpr0, 2) == round(tpr1, 2):
                results_df = results_df.append(pd.DataFrame(
                    columns=results_columns,
                    data=[('EqualOpportunity', profit_function, t0, t1) + results_group_0 + results_group_1]))

                """
                Intervention 5: Equalised Odds - uses different or equal thresholds for each category as soon as each
                group has the same rate of correctly classified loans as paid (equal TPR) AND each group has the same
                miss-classification rate of loans granted (equal FPR).
                """
                if round(fpr0, 2) == round(fpr1, 2):
                    results_df = results_df.append(pd.DataFrame(
                        columns=results_columns,
                        data=[('EqualisedOdds', profit_function, t0, t1) + results_group_0 + results_group_1]))
        print(t0, t1)

    return results_df


if __name__ == "__main__":
    df = pd.read_csv("../static/data/loans_with_predictions_df.csv", index_col=[0], low_memory=False)
    df = df.round(2)
    test_gender_male, test_gender_fem = split_test_set_by_binary_category(df, 'Gender', ['Male', 'Female'])

    gender_results_df = run_algorithmic_interventions_df(
        {'group_0': test_gender_fem, 'group_1': test_gender_male},
        {'actuals_col_name': 'PaidLoan', 'predicted_col_name': 'predicted_probabilities'},
        {'weight_tp': 1, 'weight_fp': 1.2}
    )

    gender_results_df.to_csv(
        '../static/data/fairness_measures_by_gender.csv')

    test_age_over_40, test_age_under_40 = split_test_set_by_binary_category(df, 'AgeGroup', ['Over 40', 'Under 40'])
    age_group_results_df = run_algorithmic_interventions_df(
        {'group_0': test_age_over_40, 'group_1': test_age_under_40},
        {'actuals_col_name': 'PaidLoan', 'predicted_col_name': 'predicted_probabilities'},
        {'weight_tp': 1, 'weight_fp': 2}
    )

    age_group_results_df.to_csv(
        '../static/data/fairness_measures_by_age_group.csv')

import pandas as pd
from sklearn.preprocessing import LabelBinarizer


def filter_training_columns(df_):
    training_columns = ['NewCreditCustomer', 'Amount',
                        'Interest', 'LoanDuration', 'Education',
                        'NrOfDependants', 'EmploymentDurationCurrentEmployer',
                        'IncomeFromPrincipalEmployer', 'IncomeFromPension',
                        'IncomeFromFamilyAllowance', 'IncomeFromSocialWelfare',
                        'IncomeFromLeavePay', 'IncomeFromChildSupport', 'IncomeOther',
                        'ExistingLiabilities', 'RefinanceLiabilities',
                        'DebtToIncome', 'FreeCash',
                        'CreditScoreEeMini', 'NoOfPreviousLoansBeforeLoan',
                        'AmountOfPreviousLoansBeforeLoan', 'PreviousRepaymentsBeforeLoan',
                        'PreviousEarlyRepaymentsBefoleLoan',
                        'PreviousEarlyRepaymentsCountBeforeLoan', 'PaidLoan',
                        'Council_house', 'Homeless', 'Joint_ownership', 'Joint_tenant',
                        'Living_with_parents', 'Mortgage', 'Other', 'Owner',
                        'Owner_with_encumbrance', 'Tenant', 'Entrepreneur',
                        'Fully', 'Partially', 'Retiree', 'Self_employed']

    return df_[training_columns]


def replace_columns(df_):
    new_values = {
        "NewCreditCustomer": {'Existing_credit_customer': 1, 'New_credit_Customer': 0},
        "Education": {'Higher': 5, 'Secondary': 4, 'Basic': 2, 'Vocational': 3, 'Primary': 1},
        "EmploymentDurationCurrentEmployer": {'MoreThan5Years': 6, 'UpTo3Years': 3, 'UpTo1Year': 1, 'UpTo5Years': 5,
                                              'UpTo2Years': 2, 'TrialPeriod': 0, 'UpTo4Years': 4, 'Retiree': 7,
                                              'Other': 0},
        "HomeOwnershipType": {'Tenant_unfurnished_property': 'Tenant', 'Tenant_pre_furnished_property': 'Tenant'}
    }
    return df_.replace(new_values)


def filter_rows(df_):
    df_ = df_[df_["EmploymentDurationCurrentEmployer"] != 0]
    return df_


def transform_columns_into_binary(df_, columns: list):
    for col in columns:
        lb_style = LabelBinarizer()
        lb_results = lb_style.fit_transform(df_[col])
        binary_ = pd.DataFrame(lb_results, columns=lb_style.classes_)
        df_ = pd.concat([df_, binary_], axis=1, join='inner')
    return df_


def prepare_dataframe(df_):
    df_ = replace_columns(df_)
    df_ = filter_rows(df_)
    df_ = transform_columns_into_binary(df_, ['HomeOwnershipType', 'EmploymentStatus'])
    df_ = filter_training_columns(df_)
    return df_

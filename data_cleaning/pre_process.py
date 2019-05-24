import pandas as pd
import numpy as np


def filter_rows(df_: pd.DataFrame) -> pd.DataFrame:
    """
    Returns only loans from Estonia (EE) and that are not longer running.

    :param df_: selected dataframe with loan details

    :return: EE and non-current dataframe with loans
    """
    return df_[(df_["Country"] == "EE") & (df_["Status"] != "Current")]


def filter_columns(df_: pd.DataFrame) -> pd.DataFrame:
    """
    Selects only the columns of the loans dataframe required for the project.

    :param df_: raw dataframe with loan details

    :return: dataframe with loans with selected columns
    """
    selected_columns = ['LoanNumber', 'ListedOnUTC', 'UserName', 'NewCreditCustomer',
                        'LoanDate', 'MaturityDate_Original', 'MaturityDate_Last',
                        'Age', 'DateOfBirth', 'Gender', 'Country', 'AppliedAmount',
                        'Amount', 'Interest', 'LoanDuration', 'MonthlyPayment',
                        'UseOfLoan', 'Education', 'MaritalStatus',
                        'NrOfDependants', 'EmploymentStatus', 'EmploymentDurationCurrentEmployer',
                        'WorkExperience', 'OccupationArea', 'HomeOwnershipType',
                        'IncomeFromPrincipalEmployer', 'IncomeFromPension', 'IncomeFromFamilyAllowance',
                        'IncomeFromSocialWelfare',
                        'IncomeFromLeavePay', 'IncomeFromChildSupport', 'IncomeOther',
                        'IncomeTotal', 'ExistingLiabilities', 'RefinanceLiabilities', 'DebtToIncome',
                        'FreeCash', 'DefaultDate', 'Status',
                        'CreditScoreEeMini',
                        'NoOfPreviousLoansBeforeLoan', 'AmountOfPreviousLoansBeforeLoan',
                        'PreviousRepaymentsBeforeLoan', 'PreviousEarlyRepaymentsBefoleLoan',
                        'PreviousEarlyRepaymentsCountBeforeLoan'
                        ]

    return df_[selected_columns]


def rename_columns(df_: pd.DataFrame) -> pd.DataFrame:
    """
    Renames values that should be null or numerical values that are in fact categorical.

    :param df_: selected and filtered dataframe with loan details

    :return: loan dataframe with replaced values
    """
    df_ = df_.replace(-1, np.nan)

    zero_replacements = ['Age', 'Education', 'MaritalStatus', 'EmploymentStatus', 'OccupationArea', 'CreditScoreEeMini']
    df_[zero_replacements] = df_[zero_replacements].replace(0.0, np.nan)

    value_replacements = {'UseOfLoan': {0: 'Loan_consolidation', 1: 'Real_estate', 2: 'Home_improvement', 3: 'Business',
                                        4: 'Education', 5: 'Travel', 6: 'Vehicle', 7: 'Other', 8: 'Health',
                                        101: 'Working_capital_financing', 102: 'Purchase_of_machinery_equipment',
                                        103: 'Renovation_of_real_estate', 104: 'Accounts_receivable_financing ',
                                        105: 'Acquisition_of_means_of_transport', 106: 'Construction_finance',
                                        107: 'Acquisition_of_stocks', 108: 'Acquisition_of_real_estate',
                                        109: 'Guaranteeing_obligation ', 110: 'Other_business'
                                        },
                          'Education': {1: 'Primary', 2: "Basic", 3: "Vocational", 4: "Secondary", 5: "Higher"},
                          'MaritalStatus': {1: 'Married', 2: 'Cohabitant', 3: 'Single', 4: 'Divorced', 5: 'Widow'},
                          'EmploymentStatus': {1: 'Unemployed', 2: 'Partially', 3: 'Fully', 4: 'Self_employed',
                                               5: 'Entrepreneur', 6: 'Retiree'},
                          'NewCreditCustomer': {0: 'Existing_credit_customer', 1: 'New_credit_Customer'},
                          'OccupationArea': {1: 'Other', 2: 'Mining', 3: 'Processing', 4: 'Energy', 5: 'Utilities',
                                             6: 'Construction', 7: 'Retail_and_wholesale',
                                             8: 'Transport_and_warehousing',
                                             9: 'Hospitality_and_catering', 10: 'Info_and_telecom',
                                             11: 'Finance_and_insurance', 12: 'Real_estate', 13: 'Research',
                                             14: 'Administrative', 15: 'Civil_service_and_military',
                                             16: 'Education', 17: 'Healthcare_and_social_help',
                                             18: 'Art_and_entertainment', 19: 'Agriculture_forestry_and_fishing'},
                          'HomeOwnershipType': {0: 'Homeless', 1: 'Owner', 2: 'Living_with_parents',
                                                3: 'Tenant_pre_furnished_property', 4: 'Tenant_unfurnished_property',
                                                5: 'Council_house', 6: 'Joint_tenant', 7: 'Joint_ownership',
                                                8: 'Mortgage',
                                                9: 'Owner_with_encumbrance', 10: 'Other'},
                          'NrOfDependants': {'10Plus': 11},
                          'Gender': {0: 'Male', 1: "Female", 2: "Unknown"}
                          }
    df_ = df_.replace(value_replacements)

    return df_


def add_new_columns(df_: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new columns needed for initial data exploration

    :param df_: dataframe with loan details

    :return: dataframe with loan details and new columns
    """
    df_["isLate"] = df_['Status'].apply(lambda x: 1 if x == "Late" else 0)
    df_["Defaulted"] = df_['DefaultDate'].apply(lambda x: 0 if pd.isnull(x) else 1)
    df_["DefaultStatus"] = df_["Defaulted"].apply(
        lambda x: "Did not default" if x == 0 else "Defaulted")
    df_["AgeGroup"] = df_["Age"].apply(lambda x: "Under 40" if x < 40 else "Over 40")

    return df_


def reformat_columns(df_: pd.DataFrame) -> pd.DataFrame:
    """
    R-formats the column types

    :param raw_data_df: dataframe with loan details

    :return: dataframe with loan details with reformate dcolumns
    """

    df_["ListedOnUTC"] = pd.to_datetime(df_['ListedOnUTC'])
    df_["LoanDate"] = pd.to_datetime(df_['LoanDate'])
    df_["MaturityDate_Original"] = pd.to_datetime(df_['MaturityDate_Original'])
    df_["MaturityDate_Last"] = pd.to_datetime(df_['MaturityDate_Last'])
    df_["DateOfBirth"] = pd.to_datetime(df_['DateOfBirth'])
    df_["DefaultDate"] = pd.to_datetime(df_['DefaultDate'])

    df_["LoanDuration"] = pd.to_numeric(df_["LoanDuration"])
    df_["NrOfDependants"] = pd.to_numeric(df_["NrOfDependants"])

    df_['CreditScoreEeMini'] = df_['CreditScoreEeMini'].astype(str)
    df_['isLate'] = df_['isLate'].astype(bool)
    df_['Defaulted'] = df_['Defaulted'].astype(bool)

    return df_


def pre_process_raw_data(df_: pd.DataFrame) -> pd.DataFrame:
    """
    Selects the required data and formats it as needed for the projects.

    :param df_: raw dataframe with loan details

    :return: processed dataframe with loans
    """
    df_ = filter_columns(df_)
    df_ = filter_rows(df_)
    df_ = rename_columns(df_)
    df_ = add_new_columns(df_)
    df_ = reformat_columns(df_)

    return df_

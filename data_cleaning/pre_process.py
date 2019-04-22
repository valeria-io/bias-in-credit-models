import pandas as pd
import numpy as np


def pre_process_raw_data(raw_data_df):
    """
    Formats the column types and filters loans to EE only
    :param raw_data_df: raw loan data from Bondora
    :return: original data with EE loans only and in the right column type format
    """

    df = raw_data_df.copy()

    df = df[(df["Country"] == "EE")]

    df["ListedOnUTC"] = pd.to_datetime(df['ListedOnUTC'])
    df["LoanDate"] = pd.to_datetime(df['LoanDate'])
    df["MaturityDate_Original"] = pd.to_datetime(df['MaturityDate_Original'])
    df["MaturityDate_Last"] = pd.to_datetime(df['MaturityDate_Last'])
    df["DateOfBirth"] = pd.to_datetime(df['DateOfBirth'])
    df["DefaultDate"] = pd.to_datetime(df['DefaultDate'])

    df["LoanDuration"] = pd.to_numeric(df["LoanDuration"])

    df['CreditScoreEeMini'] = df['CreditScoreEeMini'].astype(str)
    df['isLate'] = df['isLate'].astype(str)
    df['Defaulted'] = df['Defaulted'].astype(str)
    df['NrOfDependants'] = df['NrOfDependants'].astype(str)

    df["Default Status"] = df["Defaulted"].apply(
        lambda x: "Non defaulter" if x == 0 else "Defaulter")
    df["Age"] = df["Age"].apply(lambda x: "Under 40" if x <= 40 else "Over 40")

    df = df.replace(-1, np.nan)

    return df



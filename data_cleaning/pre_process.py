import pandas as pd


def pre_process_raw_data(raw_data_df):

    df = raw_data_df.copy()

    # datetime
    df["ListedOnUTC"] = pd.to_datetime(df['ListedOnUTC'])
    df["LoanDate"] = pd.to_datetime(df['LoanDate'])
    df["MaturityDate_Original"] = pd.to_datetime(df['MaturityDate_Original'])
    df["MaturityDate_Last"] = pd.to_datetime(df['MaturityDate_Last'])
    df["DateOfBirth"] = pd.to_datetime(df['DateOfBirth'])
    df["DefaultDate"] = pd.to_datetime(df['DefaultDate'])

    # numeric
    df["LoanDuration"] = pd.to_numeric(df["LoanDuration"])

    # object
    df['CreditScoreEeMini'] = df['CreditScoreEeMini'].astype(object)
    df['isLate'] = df['isLate'].astype(object)
    df['Defaulted'] = df['Defaulted'].astype(object)
    df['NrOfDependants'] = df['NrOfDependants'].astype(object)

    return df

def fill_na_with_k_nearest_neighbours():
    return -1

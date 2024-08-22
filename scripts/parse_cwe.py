import pandas as pd

global cwe_df


def load_cwe():
    global cwe_df
    cwe_df = pd.read_csv("../data/699.csv", index_col=False)


def get_df():
    return cwe_df


def id_to_cwe(cwe_id: int):
    if cwe_df is None:
        load_cwe()
    return cwe_df.loc[cwe_df["CWE-ID"] == cwe_id].to_dict(orient="records")

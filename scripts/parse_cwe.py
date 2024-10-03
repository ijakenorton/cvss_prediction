import re
from typing import Dict, List
import pandas as pd
from pprint import pprint
import utils

cwe_df = None


def load_cwe():
    global cwe_df
    cwe_df = pd.read_csv("../data/699.csv", index_col=False)


def get_df():
    return cwe_df


def id_dict_to_cwe(weakness: List[Dict]):
    global cwe_df
    if cwe_df is None:
        load_cwe()
    cwe_string = weakness[0]["description"][0]["value"]

    pattern = r"(\d+)"
    match = re.search(pattern, cwe_string)
    if match:
        cwe_id = int(match.group(1))
    else:
        return None

    return cwe_df.loc[cwe_df["CWE-ID"] == cwe_id].to_dict(orient="records")


def id_to_cwe(cwe_id: int):
    if cwe_df is None:
        load_cwe()
    return cwe_df.loc[cwe_df["CWE-ID"] == cwe_id].to_dict(orient="records")


def main():

    weaknesses = utils.read_pkl("../data/cwes.pkl")
    count = 0
    for weakness in weaknesses:
        cwe = id_dict_to_cwe(weakness)
        if cwe == None:
            count += 1

    print(count)


if __name__ == "__main__":
    main()

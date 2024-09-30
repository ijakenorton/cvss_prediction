import os
import json
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
import utils


def read_data(version_number: int):
    data = []
    ids = set()

    index = 0

    print("Parsing nvd data...")
    for _, _, files in os.walk("../nvd/"):
        for file in files:
            # print("\t", index / len(files) * 100, "%", end="\r")
            with open(f"../nvd/{file}", "r") as f:
                cves = json.load(f)

                for cve in cves:

                    desc = []
                    if f"cvssMetricV{version_number}" in cve["metrics"].keys():

                        for d in cve["descriptions"]:
                            if d["lang"] == "en":
                                desc.append(d["value"])
                        cvss_data = cve["metrics"][f"cvssMetricV{version_number}"][0][
                            "cvssData"
                        ]
                        del cvss_data["version"]
                        del cvss_data["vectorString"]

                        current = {
                            "cvssData": cve["metrics"][f"cvssMetricV{version_number}"][
                                0
                            ]["cvssData"],
                            "id": cve["id"],
                            "description": desc,
                        }

                        data.append(current)
                        ids.add(cve["id"])

            index += 1

    print("\n\t100%")
    return data, ids


def read_and_save_data(version=31):
    data, nvd_ids = read_data(version)
    nvd_data = {"data": data, "ids": nvd_ids}
    utils.write_data(nvd_data, "../data/nvd_31_cleaned.pkl")


def main():
    read_and_save_data(31)


if __name__ == "__main__":
    main()

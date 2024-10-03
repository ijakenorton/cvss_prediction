import os
import json
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
import utils


def match_cluster_words(text, cluster, min_matches=3):
    # Convert the text and cluster words to lowercase for case-insensitive matching
    text = text.lower()
    cluster_words = [word.lower() for word in cluster.split()]

    # Count how many cluster words are in the text
    matches = sum(1 for word in cluster_words if word in text)

    # Return True if the number of matches is at least min_matches
    return matches >= min_matches


def read_data(version_number: int):
    data = []
    ids = set()
    examples = []

    dates = {}
    index = 0
    weaknesses = []

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
                                if "weaknesses" in cve.keys():
                                    weaknesses.append(cve["weaknesses"])
                                else:
                                    weaknesses.append(
                                        [
                                            {
                                                "description": [
                                                    {
                                                        "lang": "en",
                                                        "value": "EMPTY",
                                                    }
                                                ],
                                                "source": "nvd@nist.gov",
                                                "type": "Primary",
                                            }
                                        ]
                                    )
                                desc.append(d["value"])
                                cluster = "site cross scripting xss plugin stored vulnerability wordpress forgery csrf"
                                if match_cluster_words(d["value"], cluster, 5):
                                    examples.append(
                                        {
                                            "id": cve["id"],
                                            "description": d["value"],
                                        }
                                    )
                                    year = cve["published"].split("-")[0]
                                    if year in dates.keys():
                                        dates[year] += 1
                                    else:
                                        dates[year] = 1

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

    # pprint(examples)
    # pprint(dates)
    # print(len(examples))

    utils.write_data(weaknesses, "../data/cwes.pkl")
    print("count weaknesses", len(weaknesses))
    print("count ids", len(ids))
    print("count data", len(data))
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

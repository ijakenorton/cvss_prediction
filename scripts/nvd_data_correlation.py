import os
import json
import utils
import pickle


def read_data():
    data = []
    ids = set()

    for _, _, files in os.walk("../nvd/"):
        for file in files:
            with open(f"../nvd/{file}", "r") as f:
                cves = json.load(f)
                for cve in cves:
                    desc = []
                    if "cvssMetricV31" in cve["metrics"].keys():
                        for d in cve["descriptions"]:
                            if d["lang"] == "en":
                                desc.append(d["value"])
                        current = {
                            "metric": cve["metrics"]["cvssMetricV31"],
                            "id": cve["id"],
                            "description": desc,
                        }

                        exists, vector_string = utils.find_key_in_nested_dict(
                            cve, "vectorString"
                        )
                        if exists:
                            current["vectorString"] = vector_string
                        data.append(current)
                        ids.add(cve["id"])

    return data, ids


# def write_data(data):
def write_data(data):
    with open("nvd_cleaned.pkl", "wb") as file:
        pickle.dump(data, file)


# write_data(read_data())

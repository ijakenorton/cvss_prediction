import os
import json
import utils
import pickle


def read_data():
    data = []
    ids = set()

    index = 0

    print("Parsing nvd data...")
    for _, _, files in os.walk("../nvd/"):
        for file in files:
            print(index / len(files) * 100, "%")
            with open(f"../nvd/{file}", "r") as f:
                cves = json.load(f)

                for cve in cves:
                    desc = []
                    if "cvssMetricV31" in cve["metrics"].keys():
                        for d in cve["descriptions"]:
                            if d["lang"] == "en":
                                desc.append(d["value"])
                        cvss_data = cve["metrics"]["cvssMetricV31"][0]["cvssData"]
                        del cvss_data["version"]
                        del cvss_data["vectorString"]
                        del cvss_data["baseSeverity"]

                        current = {
                            "cvssData": cve["metrics"]["cvssMetricV31"][0]["cvssData"],
                            "id": cve["id"],
                            "description": desc,
                        }

                        data.append(current)
                        ids.add(cve["id"])
            index += 1

    print("100%")

    return data, ids


# def write_data(data):
def write_data(data):
    with open("nvd_cleaned.pkl", "wb") as file:
        pickle.dump(data, file)


# write_data(read_data())

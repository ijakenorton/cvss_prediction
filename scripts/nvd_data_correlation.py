import os
import json
from pprint import pprint


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
                                if (
                                    "Improper access control in the firmware for some Intel(R) Processors may allow a privileged user to potentially enable a denial of service via local access."
                                    in d["value"]
                                ):
                                    print(cve["id"], "\n")
                                    print(d["value"])
                                    pprint(
                                        cve["metrics"][f"cvssMetricV{version_number}"][
                                            0
                                        ]["cvssData"]
                                    )
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
                        # if cve["id"] == "CVE-2022-21233":
                        #     print(cve)

            index += 1

    print("\n\t100%")

    return data, ids


def main():
    read_data(31)


if __name__ == "__main__":
    main()

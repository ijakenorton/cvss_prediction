import os
import json
import utils
from pprint import pprint


def read_data(version_number):
    version = ""
    match version_number:
        case 2:
            version = "2_0"
        case 30:
            version = "3_0"
        case 31:
            version = "3_1"

    data = []
    bad_vector_strings = []
    paths = []
    ids = set()
    count = 0
    for root, dirs, files in os.walk("../cvelistV5-main/cves/"):
        for dir in dirs:
            if "xxx" in dir:
                paths.append(f"{root}/{dir}")

    index = 0
    print("Parsing mitre data...")
    for path in paths[:-1]:
        print("\t", index / len(paths) * 100, "%", end="\r")
        for root, dirs, files in os.walk(path):
            for file in files:
                with open(f"{root}/{file}", "r") as f:
                    cve = json.load(f)
                    found, metric = utils.find_key_in_nested_dict(cve, "metrics")
                    _, descriptions = utils.find_key_in_nested_dict(cve, "descriptions")
                    desc = []

                    if found and metric is not None:
                        for m in metric:
                            if f"cvssV{version}" in m.keys():
                                cvss_data = parse_vector_string(
                                    m[f"cvssV{version}"]["vectorString"], version_number
                                )
                                if not cvss_data:
                                    bad_vector_strings.append(
                                        m[f"cvssV{version}"]["vectorString"]
                                    )
                                    continue
                                cvss_data["baseScore"] = m[f"cvssV{version}"][
                                    "baseScore"
                                ]

                                current = {
                                    "cvssData": cvss_data,
                                    "id": cve["cveMetadata"]["cveId"],
                                }

                                if descriptions is not None:
                                    for d in descriptions:
                                        if d["lang"] == "en":
                                            desc.append(d["value"])
                                            current["description"] = desc

                                data.append(current)

                                ids.add(cve["cveMetadata"]["cveId"])
        index += 1
    print("\n\t100%")
    # print(bad_vector_strings)
    # print(count)

    return data, ids


def parse_vector_string(vector_string: str, version):
    terms = vector_string.split("/")

    corrected_terms = []
    for term in terms:
        pairs = term.split(":")
        metric, dimension = utils.metric_mappings(pairs[0], pairs[1], version)
        if not metric or not dimension:
            continue
        corrected_term = [metric, dimension]
        corrected_terms.append(corrected_term)
    corrected = dict(corrected_terms)
    return corrected

import os
import json
import utils
import pickle


def read_data():
    data = []
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
        print(index / len(paths) * 100, "%")
        for root, dirs, files in os.walk(path):
            for file in files:
                with open(f"{root}/{file}", "r") as f:
                    cve = json.load(f)
                    found, metric = utils.find_key_in_nested_dict(cve, "metrics")
                    _, descriptions = utils.find_key_in_nested_dict(cve, "descriptions")
                    desc = []

                    if found and metric is not None:
                        for m in metric:
                            if "cvssV3_1" in m.keys():
                                cvss_data = parse_vector_string(
                                    m["cvssV3_1"]["vectorString"]
                                )
                                if not cvss_data:
                                    count += 1
                                    continue
                                cvss_data["baseScore"] = m["cvssV3_1"]["baseScore"]

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

    print("100%")
    # print(count)
    return data, ids


def write_data(data):
    with open("mitre_cleaned.pkl", "wb") as file:
        pickle.dump(data, file)


def parse_vector_string(vector_string: str):
    terms = vector_string.split("/")[1:]
    corrected_terms = []
    for term in terms:
        pairs = term.split(":")

        # metric = utils.map_metrics(pairs[0])
        # dimension = utils.map_dimensions(pairs[1])
        metric, dimension = utils.metric_mappings(pairs[0], pairs[1])
        if not metric or not dimension:
            return None

        corrected_term = [metric, dimension]

        corrected_terms.append(corrected_term)
    return dict(corrected_terms)


# write_data(read_data())

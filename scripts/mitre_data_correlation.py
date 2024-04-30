import os
import json
import utils
import pickle


def read_data():
    data = []
    paths = []
    ids = set()
    for root, dirs, files in os.walk("../cvelistV5-main/cves/"):
        for dir in dirs:
            if "xxx" in dir:
                paths.append(f"{root}/{dir}")

    count = 0
    for index, path in enumerate(paths[:-1]):
        # print(index / len(paths) * 100, "%")
        for root, dirs, files in os.walk(path):
            for file in files:
                with open(f"{root}/{file}", "r") as f:
                    cve = json.load(f)
                    # found, cvss = utils.find_key_in_nested_dict(cve, "cvssV3_1")

                    found, cvss = utils.find_key_in_nested_dict(cve, "baseScore")
                    if found:
                        count += 1

                    # found, metric = utils.find_key_in_nested_dict(cve, "metrics")
                    # _, descriptions = utils.find_key_in_nested_dict(cve, "descriptions")
                    # desc = []

                    # if found and metric is not None:
                    #     for m in metric:
                    #         if "cvssV3_1" in m.keys():
                    #             is_there, vector_string = utils.find_key_in_nested_dict(
                    #                 cve, "vectorString"
                    #             )
                    #             if is_there:
                    #                 print(vector_string)
                    #             count += 1
                    #             current = {
                    #                 "metric": m["cvssV3_1"],
                    #                 "name": cve["cveMetadata"]["cveId"],
                    #             }
                    #             if descriptions is not None:
                    #                 for d in descriptions:
                    #                     if d["lang"] == "en":
                    #                         desc.append(d["value"])

                    #             data.append(current)
                    #             ids.add(cve["cveMetadata"]["cveId"])

                    # is_there, vector_string = find_key_in_nested_dict(
                    #     cve, "vectorString"
                    # )
                    # if is_there:
                    #     vector_string_count += 1

                    # print("no en", vector_string)
                    # print("metric", m["cvssV3_1"])

    print("count", count)
    # print(vector_string_count)

    return data, ids


def write_data(data):
    with open("mitre_cleaned.pkl", "wb") as file:
        pickle.dump(data, file)


# write_data(read_data())

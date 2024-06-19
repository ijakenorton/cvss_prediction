#!/home/jaken/miniconda3/envs/bayes/bin/python

import nvd_data_correlation
import mitre_data_correlation
import utils
import inspect
import argparse


def parse(source: str, version: int):
    match source:
        case "nvd":
            data, nvd_ids = nvd_data_correlation.read_data(version)
            utils.write_data(data, f"../data/nvd_{version}_cleaned.pkl")
            nvd_data = {"data": data, "ids": nvd_ids}
            return nvd_data
        case "mitre":
            data, mitre_ids = mitre_data_correlation.read_data(version)
            utils.write_data(data, f"../data/mitre_{version}_cleaned.pkl")
            mitre_data = {"data": data, "ids": mitre_ids}
            return mitre_data
        case _:
            print("Incorrect source")


# description,av,ac,pr,ui,s,c,i,a

# dict_keys(['attackVector', 'attackComplexity', 'privilegesRequired', 'userInteraction', 'scope', 'confidentialityImpact', 'integrityImpact', 'availabilityImpact', 'baseScore', 'baseSeverity'])


# def check(source: str, version: int) -> None:
#     data = utils.read_data(f"../data/{source}_{version}_cleaned.pkl")
#     all_keys = set()

#     for d in data["data"]:
#         keys = d["cvssData"].keys()
#         print(keys)
#         exit()
#         for key in keys:
#             all_keys.add(key)

#     print(all_keys)


def make_lines(data):
    output = ""
    for d in data["data"]:
        if "description" not in d.keys():
            continue

        outstr = (
            d["description"][0].replace("\n", "").replace("\r", "").replace(",", " ")
        )
        cvssData = d["cvssData"]
        for key in cvssData:
            if key in ["baseScore", "baseSeverity"]:
                continue
            outstr = f"{outstr},{cvssData[key]}"
        output += outstr + "\n"
    return output


def make_combined_csv(version: int) -> None:
    nvd_data = utils.read_data(f"../data/nvd_{version}_cleaned.pkl")
    mitre_data = utils.read_data(f"../data/mitre_{version}_cleaned.pkl")

    with open(f"../data/combined_{version}.csv", "w") as f:
        f.write(f"{make_lines(nvd_data)}")
        f.write(f"{make_lines(mitre_data)}\n")


def make_csv_with_checks(source: str, version: int) -> None:
    data = utils.read_data(f"../data/{source}_{version}_cleaned.pkl")
    count = 0
    multiple = 0
    with open(f"../data/{source}_{version}.csv", "w") as f:
        for d in data["data"]:
            if "description" not in d.keys():
                count += 1
                continue
            if len(d["description"]) != 1:
                multiple += 1
                print(d["cvssData"])

            outstr = (
                d["description"][0]
                .replace("\n", "")
                .replace("\r", "")
                .replace(",", " ")
            )
            cvssData = d["cvssData"]
            for key in cvssData:
                if key in ["baseScore", "baseSeverity"]:
                    continue
                outstr = f"{outstr}, {cvssData[key]}"
            f.write(f"{outstr}\n")
    print("count:", count)
    print("multiple:", multiple)


def main():
    make_combined_csv(31)
    exit()

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, help="CVSS version to parse")
    args = parser.parse_args()
    version = args.version
    if version is not None:
        nvd_data = parse("nvd", version)
        mitre_data = parse("mitre", version)


if __name__ == "__main__":
    main()

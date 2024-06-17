#!/home/jaken/miniconda3/envs/bayes/bin/python

import nvd_data_correlation
import mitre_data_correlation
import utils
import argparse


def parse(source, version):
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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, help="CVSS version to parse")
    args = parser.parse_args()
    version = args.version
    if version is not None:
        nvd_data = parse("nvd", version)
        mitre_data = parse("mitre", version)


if __name__ == "__main__":
    main()

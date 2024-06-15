import nvd_data_correlation
import mitre_data_correlation
import utils
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, help="CVSS version to parse")
    args = parser.parse_args()
    version = args.version
    if version is not None:
        data, nvd_ids = nvd_data_correlation.read_data(version)
        nvd_data = {"data": data, "ids": nvd_ids}
        utils.write_data(nvd_data, f"../data/nvd_{version}_cleaned.pkl")
        # data, mitre_ids = mitre_data_correlation.read_data(version)
        # mitre_data = {"data": data, "ids": mitre_ids}
        # utils.write_data(mitre_data, f"../data/mitre_{version}_cleaned_..pkl")


if __name__ == "__main__":
    main()

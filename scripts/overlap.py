import nvd_data_correlation
import mitre_data_correlation
import metric_counts
import plot
from cvss_types import Metrics
import utils


def overlap():
    mitre_result = utils.read_data("../data/mitre_cleaned.pkl")
    nvd_result = utils.read_data("../data/nvd_cleaned.pkl")

    mitre_data, mitre_ids = mitre_result["data"], mitre_result["ids"]
    nvd_data, nvd_ids = nvd_result["data"], nvd_result["ids"]

    overlap = set()
    count = 0
    for id in nvd_ids:
        if id in mitre_ids:
            count += 1
            overlap.add(id)

    nvd_overlap = [cve for cve in nvd_data if cve["id"] in overlap]
    mitre_overlap = [cve for cve in mitre_data if cve["id"] in overlap]
    return nvd_overlap, mitre_overlap


def main():
    data, nvd_ids = nvd_data_correlation.read_data()
    nvd_data = {"data": data, "ids": nvd_ids}
    data, mitre_ids = mitre_data_correlation.read_data()
    mitre_data = {"data": data, "ids": mitre_ids}
    print("writing data to file....")

    utils.write_data(nvd_data, "../data/nvd_2_cleaned.pkl")
    utils.write_data(mitre_data, "../data/mitre_2_cleaned_..pkl")

    # plot.plot_metrics(nvd_metrics)
    # plot.plot_metrics(mitre_metrics)
    # nvd_overlap, mitre_overlap = overlap()

    # plot.plot_comparison(nvd_overlap, mitre_overlap)


if __name__ == "__main__":
    main()

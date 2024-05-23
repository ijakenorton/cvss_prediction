import nvd_data_correlation
import mitre_data_correlation
import metric_counts
import plot
import utils


def main():
    # data, nvd_ids = nvd_data_correlation.read_data()
    # nvd_data = {"data": data, "ids": nvd_ids}
    # data, mitre_ids = mitre_data_correlation.read_data()
    # mitre_data = {"data": data, "ids": mitre_ids}
    # print("writing data to file....")

    # utils.write_data(nvd_data, "../data/nvd_cleaned.pkl")
    # utils.write_data(mitre_data, "../data/mitre_cleaned.pkl")

    # Read data from file and store in a dictionary
    mitre_result = utils.read_data("../data/mitre_cleaned.pkl")
    nvd_result = utils.read_data("../data/nvd_cleaned.pkl")

    # Unpack values from the dictionary into variables
    mitre_data, mitre_ids = mitre_result["data"], mitre_result["ids"]
    nvd_data, nvd_ids = nvd_result["data"], nvd_result["ids"]

    print(len(nvd_ids))
    print(len(mitre_ids))
    overlap = set()
    count = 0
    for id in nvd_ids:
        if id in mitre_ids:
            count += 1
            overlap.add(id)

    nvd_overlap = [cve for cve in nvd_data if cve["id"] in overlap]
    mitre_overlap = [cve for cve in mitre_data if cve["id"] in overlap]
    print(len(nvd_overlap))
    print(len(mitre_overlap))
    nvd_metrics = metric_counts.calculate_metric_counts(nvd_overlap)
    mitre_metrics = metric_counts.calculate_metric_counts(mitre_overlap)

    for key in nvd_metrics:
        print(nvd_metrics[key])

    for key in mitre_metrics:
        print(mitre_metrics[key])

    plot.plot_metrics(nvd_metrics)
    plot.plot_metrics(mitre_metrics)


if __name__ == "__main__":
    main()

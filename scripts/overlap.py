import nvd_data_correlation
import mitre_data_correlation
import utils


def main():
    # data, nvd_ids = nvd_data_correlation.read_data()
    # nvd_data = {"data": data, "ids": nvd_ids}
    data, mitre_ids = mitre_data_correlation.read_data()
    mitre_data = {"data": data, "ids": mitre_ids}
    print("writing data to file....")
    # utils.write_data(nvd_data, "../data/nvd_cleaned.pkl")
    utils.write_data(mitre_data, "../data/mitre_cleaned.pkl")
    # print(len(mitre_ids))

    # count = 0
    # for id in nvd_ids:
    #     if id in mitre_ids:
    #         count += 1
    #         print(id)
    # print(count)


if __name__ == "__main__":
    main()

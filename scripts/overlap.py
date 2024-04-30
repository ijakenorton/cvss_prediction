import nvd_data_correlation
import mitre_data_correlation


def main():
    # data, nvd_ids = nvd_data_correlation.read_data()
    # nvd_data_correlation.write_data(data)
    data, mitre_ids = mitre_data_correlation.read_data()
    print(len(mitre_ids))

    # count = 0
    # for id in nvd_ids:
    #     if id in mitre_ids:
    #         count += 1
    #         print(id)
    # print(count)


if __name__ == "__main__":
    main()

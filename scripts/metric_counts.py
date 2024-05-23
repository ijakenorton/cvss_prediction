def metrics_counts():
    metrics = {
        "attackVector": {
            "NETWORK": 0,
            "ADJACENT_NETWORK": 0,
            "LOCAL": 0,
            "PHYSICAL": 0,
        },
        "attackComplexity": {
            "LOW": 0,
            "HIGH": 0,
        },
        "privilegesRequired": {
            "NONE": 0,
            "LOW": 0,
            "HIGH": 0,
        },
        "userInteraction": {
            "NONE": 0,
            "REQUIRED": 0,
        },
        "scope": {
            "UNCHANGED": 0,
            "CHANGED": 0,
        },
        "confidentialityImpact": {
            "NONE": 0,
            "LOW": 0,
            "HIGH": 0,
        },
        "integrityImpact": {
            "NONE": 0,
            "LOW": 0,
            "HIGH": 0,
        },
        "availabilityImpact": {
            "NONE": 0,
            "LOW": 0,
            "HIGH": 0,
        },
    }
    return metrics


def calculate_metric_counts(data):
    metrics = metrics_counts()
    for cve in data:
        for metric, value in dict.items(cve["cvssData"]):
            if metric == "baseScore":
                continue
            # if value not in cve["cvssData"].keys():
            #     continue

            metrics[metric][value] += 1
    return metrics


# if __name__ == "__main__":

# for key in calculate_metric_counts():
#     print(metrics_counts[key])

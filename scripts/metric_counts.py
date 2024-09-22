from cvss_types import Metrics
from pprint import pprint


def v2_metrics_counts():
    metrics = {
        "accessVector": {
            "ADJACENT_NETWORK": 0,
            "LOCAL": 0,
            "NETWORK": 0,
        },
        "accessComplexity": {
            "LOW": 0,
            "MEDIUM": 0,
            "HIGH": 0,
        },
        "authentication": {
            "MULTIPLE": 0,
            "SINGLE": 0,
            "NONE": 0,
        },
        "confidentialityImpact": {
            "NONE": 0,
            "PARTIAL": 0,
            "COMPLETE": 0,
        },
        "integrityImpact": {
            "NONE": 0,
            "PARTIAL": 0,
            "COMPLETE": 0,
        },
        "availabilityImpact": {
            "NONE": 0,
            "PARTIAL": 0,
            "COMPLETE": 0,
        },
    }
    return metrics


def v3_metrics_counts() -> Metrics:
    metrics: Metrics = {
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


def calculate_metric_counts(data, version, field="cvssData"):
    metrics = v2_metrics_counts() if version == 2 else v3_metrics_counts()

    for cve in data:
        if field not in cve.keys():
            continue
        for metric, value in dict.items(cve[field]):
            if metric == "baseScore":
                continue
            if metric == "baseSeverity":
                continue

            metrics[metric][value] += 1
    return metrics

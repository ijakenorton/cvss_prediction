import pickle
from typing import Dict, TypedDict, List, Any


class CvssData(TypedDict):
    data: List[Dict[str, Any]]
    ids: List[str]


v2_dimensions = {
    "accessVector": {
        "a": "ADJACENT_NETWORK",
        "l": "LOCAL",
        "n": "NETWORK",
    },
    "accessComplexity": {
        "l": "LOW",
        "m": "MEDIUM",
        "h": "HIGH",
    },
    "authentication": {
        "m": "MULTIPLE",
        "s": "SINGLE",
        "n": "NONE",
    },
    "confidentialityImpact": {
        "n": "NONE",
        "p": "PARTIAL",
        "c": "COMPLETE",
    },
    "integrityImpact": {
        "n": "NONE",
        "p": "PARTIAL",
        "c": "COMPLETE",
    },
    "availabilityImpact": {
        "n": "NONE",
        "p": "PARTIAL",
        "c": "COMPLETE",
    },
}

v3_dimensions = {
    "attackVector": {
        "a": "ADJACENT_NETWORK",
        "l": "LOCAL",
        "n": "NETWORK",
        "p": "PHYSICAL",
    },
    "attackComplexity": {
        "l": "LOW",
        "h": "HIGH",
    },
    "privilegesRequired": {
        "n": "NONE",
        "l": "LOW",
        "h": "HIGH",
    },
    "userInteraction": {
        "n": "NONE",
        "r": "REQUIRED",
    },
    "scope": {
        "u": "UNCHANGED",
        "c": "CHANGED",
    },
    "confidentialityImpact": {
        "n": "NONE",
        "l": "LOW",
        "h": "HIGH",
    },
    "integrityImpact": {
        "n": "NONE",
        "l": "LOW",
        "h": "HIGH",
    },
    "availabilityImpact": {
        "n": "NONE",
        "l": "LOW",
        "h": "HIGH",
    },
}


def metrics_v2(metric):
    match metric:
        case "av":
            return "accessVector"
        case "ac":
            return "accessComplexity"
        case "au":
            return "authentication"
        case "c":
            return "confidentialityImpact"
        case "i":
            return "integrityImpact"
        case "a":
            return "availabilityImpact"
        case _:
            return None


def metrics_v3(metric):
    match metric:
        case "av":
            return "attackVector"
        case "ac":
            return "attackComplexity"
        case "pr":
            return "privilegesRequired"
        case "ui":
            return "userInteraction"
        case "s":
            return "scope"
        case "c":
            return "confidentialityImpact"
        case "i":
            return "integrityImpact"
        case "a":
            return "availabilityImpact"
        case _:
            return None


def dimensions_v2(metric, dimension):
    if (
        metric not in v2_dimensions.keys()
        or dimension not in v2_dimensions[metric].keys()
    ):
        return None, None

    return metric, v2_dimensions[metric][dimension]


def dimensions_v3(metric, dimension):
    if (
        metric not in v3_dimensions.keys()
        or dimension not in v3_dimensions[metric].keys()
    ):
        return None, None

    return metric, v3_dimensions[metric][dimension]


def dimensions(metric, version):
    return (
        list(v2_dimensions[metric].values())
        if version == 2
        else list(v3_dimensions[metric].values())
    )


def inverse_dimension_mappings():
    return {
        "ADJACENT_NETWORK": "A",
        "CHANGED": "C",
        "COMPLETE": "C",
        "HIGH": "H",
        "LOW": "L",
        "LOCAL": "L",
        "MEDIUM": "M",
        "MULTIPLE": "M",
        "NONE": "N",
        "NETWORK": "N",
        "PHYSICAL": "P",
        "PARTIAL": "P",
        "REQUIRED": "R",
        "SINGLE": "S",
        "UNCHANGED": "U",
    }


def inverse_from_keys(keys):
    inverse = []
    for key in keys:
        inverse.append(inverse_dimension_mappings()[key])
    return inverse


def metric_mappings(metric, dimension, version):
    metric = metric.lower()
    dimension = dimension.lower()
    metric = metrics_v2(metric) if version == 2 else metrics_v3(metric)

    if metric == None:
        return None, None

    return (
        dimensions_v2(metric, dimension)
        if version == 2
        else dimensions_v3(metric, dimension)
    )


def find_key_in_nested_dict(dictionary, target_key):
    if target_key in dictionary:
        return True, dictionary[target_key]

    if isinstance(dictionary, dict):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                found, result = find_key_in_nested_dict(value, target_key)
                if found:
                    return True, result
            if isinstance(value, list):
                for v in value:
                    found, result = find_key_in_nested_dict(v, target_key)
                    if found:
                        return True, result

    return False, None


def filter_by_metric_score(data, metric, score):

    data = list(filter(lambda x: x["cvssData"][metric] == score, data))
    return data


def ids(version=31):

    nvd_data = read_data(f"../data/nvd_{version}_cleaned.pkl")
    data = nvd_data["data"]
    ids = list(map(lambda x: x["id"], data))
    return ids


def write_data(data, path):
    with open(path, "wb") as file:
        pickle.dump(data, file)


def read_pkl(path):
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data


def read_data(path) -> CvssData:
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data

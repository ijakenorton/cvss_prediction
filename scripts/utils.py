import pickle
from typing import Dict, TypedDict, List, Any


class CvssData(TypedDict):
    data: List[Dict[str, Any]]
    ids: List[str]


v2_dimensions = {
    "accessVector": {
        "A": "ADJACENT_NETWORK",
        "L": "LOCAL",
        "N": "NETWORK",
    },
    "accessComplexity": {
        "L": "LOW",
        "M": "MEDIUM",
        "H": "HIGH",
    },
    "authentication": {
        "M": "MULTIPLE",
        "S": "SINGLE",
        "N": "NONE",
    },
    "confidentialityImpact": {
        "N": "NONE",
        "P": "PARTIAL",
        "C": "COMPLETE",
    },
    "integrityImpact": {
        "N": "NONE",
        "P": "PARTIAL",
        "C": "COMPLETE",
    },
    "availabilityImpact": {
        "N": "NONE",
        "P": "PARTIAL",
        "C": "COMPLETE",
    },
}

v3_dimensions = {
    "attackVector": {
        "A": "ADJACENT_NETWORK",
        "L": "LOCAL",
        "N": "NETWORK",
        "P": "PHYSICAL",
    },
    "attackComplexity": {
        "L": "LOW",
        "H": "HIGH",
    },
    "privilegesRequired": {
        "N": "NONE",
        "L": "LOW",
        "H": "HIGH",
    },
    "userInteraction": {
        "N": "NONE",
        "R": "REQUIRED",
    },
    "scope": {
        "U": "UNCHANGED",
        "C": "CHANGED",
    },
    "confidentialityImpact": {
        "N": "NONE",
        "L": "LOW",
        "H": "HIGH",
    },
    "integrityImpact": {
        "N": "NONE",
        "L": "LOW",
        "H": "HIGH",
    },
    "availabilityImpact": {
        "N": "NONE",
        "L": "LOW",
        "H": "HIGH",
    },
}


def metrics_v2(metric):
    match metric:
        case "AV":
            return "accessVector"
        case "AC":
            return "accessComplexity"
        case "Au":
            return "authentication"
        case "C":
            return "confidentialityImpact"
        case "I":
            return "integrityImpact"
        case "A":
            return "availabilityImpact"
        case _:
            return None


def metrics_v3(metric):
    match metric:
        case "AV":
            return "attackVector"
        case "AC":
            return "attackComplexity"
        case "PR":
            return "privilegesRequired"
        case "UI":
            return "userInteraction"
        case "S":
            return "scope"
        case "C":
            return "confidentialityImpact"
        case "I":
            return "integrityImpact"
        case "A":
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


def write_data(data, path):
    with open(path, "wb") as file:
        pickle.dump(data, file)


def read_data(path) -> CvssData:
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data

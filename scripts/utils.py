import pickle
from typing import Dict, TypedDict, List, Any


class CvssData(TypedDict):
    data: List[Dict[str, Any]]
    ids: List[str]


def dimension_mappings():
    return {
        "A": "ADJACENT_NETWORK",
        "L": "LOW",
        "N": "NONE",
        "H": "HIGH",
        "R": "REQUIRED",
        "U": "UNCHANGED",
        "C": "CHANGED",
        "P": "PHYSICAL",
    }


def inverse_dimension_mappings():
    return {
        "ADJACENT_NETWORK": "A",
        "LOW": "L",
        "LOCAL": "L",
        "NONE": "N",
        "NETWORK": "N",
        "HIGH": "H",
        "REQUIRED": "R",
        "UNCHANGED": "U",
        "CHANGED": "C",
        "PHYSICAL": "P",
    }


def inverse_from_keys(keys):
    inverse = []
    for key in keys:
        inverse.append(inverse_dimension_mappings()[key])
    return inverse


def metrics_and_dimensions():
    return {
        "attackVector": {
            "N": "NETWORK",
            "A": "ADJACENT_NETWORK",
            "L": "LOCAL",
            "P": "PHYSICAL",
        },
        "attackComplexity": dimension_mappings(),
        "privilegesRequired": dimension_mappings(),
        "userInteraction": dimension_mappings(),
        "scope": dimension_mappings(),
        "confidentialityImpact": dimension_mappings(),
        "integrityImpact": dimension_mappings(),
        "availabilityImpact": dimension_mappings(),
    }


def metrics():
    return [
        "attackVector",
        "attackComplexity",
        "privilegesRequired",
        "userInteraction",
        "scope",
        "confidentialityImpact",
        "integrityImpact",
        "availabilityImpact",
    ]


def metric_mappings(metric, dimension):
    metric_mapping = {
        "AV": "attackVector",
        "AC": "attackComplexity",
        "PR": "privilegesRequired",
        "UI": "userInteraction",
        "S": "scope",
        "C": "confidentialityImpact",
        "I": "integrityImpact",
        "A": "availabilityImpact",
    }
    if metric not in metric_mapping.keys():
        return None, None
    metric = metric_mapping[metric]
    dimension = metrics_and_dimensions()[metric][dimension]
    return metric, dimension


# def map_dimensions(short):
#     if short not in dimension_mappings().keys():
#         return None

#     return dimension_mappings()[short]


# def map_metrics(short):
#     if short not in metric_mappings().keys():
#         return None

#     return metric_mappings()[short]


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
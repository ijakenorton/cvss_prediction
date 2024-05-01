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


def metric_mappings():
    return {
        "AV": "attackVector",
        "AC": "attackComplexity",
        "PR": "privilegesRequired",
        "UI": "userInteraction",
        "S": "scope",
        "C": "confidentialityImpact",
        "I": "integrityImpact",
        "A": "availabilityImpact",
    }


def dimension_mappings():
    return {
        "A": "ADJACENT",
        "L": "LOW",
        "N": "NONE",
        "H": "HIGH",
        "R": "REQUIRED",
        "U": "UNCHANGED",
        "C": "CHANGED",
    }


def map_dimensions(short):
    if short not in dimension_mappings().keys():
        return None

    return dimension_mappings()[short]


def map_metrics(short):
    if short not in metric_mappings().keys():
        return None

    return metric_mappings()[short]


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

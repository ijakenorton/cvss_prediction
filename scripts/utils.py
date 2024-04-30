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

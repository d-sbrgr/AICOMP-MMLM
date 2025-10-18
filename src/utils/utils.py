def unflatten_config(d, sep="."):
    def merge(a, b):
        for k, v in b.items():
            if k in a and isinstance(a[k], dict) and isinstance(v, dict):
                merge(a[k], v)
            else:
                a[k] = v
        return a

    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            # Recurse for already nested dicts
            value = unflatten_config(value, sep)
        if sep in key:
            parts = key.split(sep)
            nested = current = {}
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = value
            result = merge(result, nested)
        else:
            result = merge(result, {key: value})
    return result

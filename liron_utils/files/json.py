import pandas as pd


def load_json(file):
    # TODO replace with pd.read_json
    import json

    with open(file, "rb") as f:
        d = json.load(f)
    return d


def write_json(d, file, *args, **kwargs):
    kwargs = dict(indent=4) | kwargs
    # dump = json.dumps(d, *args, **kwargs)
    # with open(file, 'w') as f:
    #     json.dump(dump, f)
    pd.Series(d).to_json(file, *args, **kwargs)

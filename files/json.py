import json


def load_json(file):
    with open(file) as f:
        cfg = json.load(f)
    return cfg

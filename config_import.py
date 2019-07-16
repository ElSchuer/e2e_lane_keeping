import json

def load_config(filename):
    data = dict()
    with open(filename) as json_file:
        data = json.load(json_file)

    return data
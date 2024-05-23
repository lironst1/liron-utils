import os
import json


def _add_json_ext(file):
	if len(os.path.splitext(file)[-1]) == 0:
		file += '.json'
	return file


def load_json(file):
	file = _add_json_ext(file)

	with open(file) as f:
		d = json.load(f)
	return d


def write_json(d, file):
	file = _add_json_ext(file)

	with open(file, 'w', encoding='utf-8') as f:
		json.dump(d, f, ensure_ascii=False, indent=4)

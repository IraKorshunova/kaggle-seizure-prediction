import json

with open('SETTINGS.json') as data_file:
    paths = json.load(data_file)

METADATA_PATH = paths["METADATA_PATH"]
RAW_DATA_PATH = paths["RAW_DATA_PATH"]
PROCESSED_DATA_PATH = paths["PROCESSED_DATA_PATH"]
LABELS_PATH = paths["LABELS_PATH"]
IMG_PATH = paths["IMG_PATH"]
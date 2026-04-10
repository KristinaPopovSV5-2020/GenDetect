import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    def __init__(self):
        self.INPUT_DATA_FOLDER = os.getenv("INPUT_DATA_FOLDER")
        self.OUTPUT_DATASET_FOLDER = os.getenv("OUTPUT_DATASET_FOLDER")

        if self.INPUT_DATA_FOLDER is None:
            raise ValueError("INPUT_DATA_FOLDER is not set")

        if self.OUTPUT_DATASET_FOLDER is None:
            raise ValueError("OUTPUT_DATASET_FOLDER is not set")


config = Config()
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    def __init__(self):
        self.INPUT_DATA_FOLDER = os.getenv("INPUT_DATA_FOLDER")
        self.OUTPUT_DATASET_FOLDER = os.getenv("OUTPUT_DATASET_FOLDER")
        self.MODEL_FOLDER = os.getenv("MODEL_FOLDER")

        if not self.INPUT_DATA_FOLDER:
            raise ValueError("INPUT_DATA_FOLDER is not set")

        if not self.OUTPUT_DATASET_FOLDER:
            raise ValueError("OUTPUT_DATASET_FOLDER is not set")

        self.TRAIN_DIR = os.path.join(self.OUTPUT_DATASET_FOLDER, "train")
        self.VAL_DIR = os.path.join(self.OUTPUT_DATASET_FOLDER, "val")
        self.TEST_DIR = os.path.join(self.OUTPUT_DATASET_FOLDER, "test")

config = Config()
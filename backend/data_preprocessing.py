import os
import shutil
import glob
from sklearn.model_selection import train_test_split

from env_config import config


def train_val_test_split(data, random_state=42):
    x = [x[0] for x in data]
    y = [x[1] for x in data]  # real or fake

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state, stratify=y
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=0.125, random_state=random_state, stratify=y_train_val
    )

    train = list(zip(x_train, y_train))
    val = list(zip(x_val, y_val))
    test = list(zip(x_test, y_test))

    return train, val, test



def load_image_paths(data_folder, limit_per_class=8000):
    data = []
    for label in ['real', 'fake']:
        folder = os.path.join(data_folder, label)

        filepaths = sorted(glob.glob(os.path.join(folder, '*')))
        filepaths = filepaths[:limit_per_class]

        for filepath in filepaths:
            data.append((filepath, label))

    return data


def save_split(data_split, split_name, output_root='data'):
    for filepath, label in data_split:
        dest_dir = os.path.join(output_root, split_name, label)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(filepath, dest_dir)


def organize_dataset(original_data_folder,output_folder):
    data = load_image_paths(original_data_folder)
    train, val, test = train_val_test_split(data)

    save_split(train, 'train', output_folder)
    save_split(val, 'val', output_folder)
    save_split(test, 'test', output_folder)


if __name__ == '__main__':
    organize_dataset(
        config.INPUT_DATA_FOLDER,
        config.OUTPUT_DATASET_FOLDER
    )
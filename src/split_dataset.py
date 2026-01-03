import os
import shutil
import random
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)

RAW_DIR = "data/raw/NEU-DET"
OUT_DIR = "data/processed"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

os.makedirs(OUT_DIR, exist_ok=True)

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUT_DIR, split), exist_ok=True)

for class_name in os.listdir(RAW_DIR):
    class_path = os.path.join(RAW_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    train_imgs, temp_imgs = train_test_split(
        images, test_size=(1 - TRAIN_RATIO), random_state=SEED
    )
    val_imgs, test_imgs = train_test_split(
        temp_imgs,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=SEED
    )

    for split, imgs in zip(
        ["train", "val", "test"],
        [train_imgs, val_imgs, test_imgs]
    ):
        split_class_dir = os.path.join(OUT_DIR, split, class_name)
        os.makedirs(split_class_dir, exist_ok=True)

        for img in imgs:
            shutil.copy(
                os.path.join(class_path, img),
                os.path.join(split_class_dir, img)
            )

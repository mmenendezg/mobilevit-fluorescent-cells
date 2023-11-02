import os
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import DataLoader
import albumentations as A
import pytorch_lightning as pl
from transformers import AutoImageProcessor
from datasets import Dataset, DatasetDict

# Checkpoint of the model used in the projec
MODEL_CHECKPOINT = "apple/deeplabv3-mobilevit-xx-small"
# UPDATE THIS IN THE FINAL MAIN FILE
RAW_DATA_PATH = "../data/raw/"
# Size of the image used to train the model
IMG_SIZE = [256, 256]


class FluorescentNeuronalDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir=RAW_DATA_PATH, dataset_size=1.0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_processor = AutoImageProcessor.from_pretrained(
            MODEL_CHECKPOINT, do_reduce_labels=False
        )
        self.image_resizer = A.Compose(
            [
                A.Resize(
                    width=IMG_SIZE[0],
                    height=IMG_SIZE[1],
                    interpolation=cv2.INTER_NEAREST,
                )
            ]
        )
        self.image_augmentator = A.Compose(
            [
                A.HorizontalFlip(p=0.6),
                A.VerticalFlip(p=0.6),
                A.RandomBrightnessContrast(p=0.6),
                A.RandomGamma(p=0.6),
                A.HueSaturationValue(p=0.6),
            ]
        )

        # Percentage of the dataset
        self.dataset_size = dataset_size

    def _create_dataset(self):
        images_path = os.path.join(self.data_dir, "all_images", "images")
        masks_path = os.path.join(self.data_dir, "all_masks", "masks")
        list_images = os.listdir(images_path)

        # Determine the size of the dataset
        if self.dataset_size < 1.0:
            n_images = int(len(list_images) * self.dataset_size)
            list_images = list_images[:n_images]

        images = []
        masks = []
        for image_filename in list_images:
            image_path = os.path.join(images_path, image_filename)
            mask_path = os.path.join(masks_path, image_filename)

            image = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
            mask = (mask / 255).astype(np.uint8)

            images.append(image)
            masks.append(mask)

        dataset = Dataset.from_dict({"image": images, "mask": masks})

        # Split the dataset into train, val, and test sets
        dataset = dataset.train_test_split(test_size=0.1)
        train_val = dataset["train"]
        test_ds = dataset["test"]
        del dataset

        train_val = train_val.train_test_split(test_size=0.2)
        train_ds = train_val["train"]
        valid_ds = train_val["test"]
        del train_val

        dataset = DatasetDict(
            {"train": train_ds, "validation": valid_ds, "test": test_ds}
        )
        del train_ds, valid_ds, test_ds
        return dataset

    def _transform_train_data(self, batch):
        # Preprocess the images
        images, masks = [], []
        for i, m in zip(batch["image"], batch["mask"]):
            img = np.asarray(i, dtype=np.uint8)
            mask = np.asarray(m, dtype=np.uint8)
            # First resize the images and masks
            resized_outputs = self.image_resizer(image=img, mask=mask)
            images.append(resized_outputs["image"])
            masks.append(resized_outputs["mask"])

            # Then augment the images
            augmented_outputs = self.image_augmentator(
                image=resized_outputs["image"], mask=resized_outputs["mask"]
            )
            images.append(augmented_outputs["image"])
            masks.append(augmented_outputs["mask"])

        inputs = self.image_processor(
            images=images,
            return_tensors="pt",
        )
        inputs["labels"] = torch.tensor(masks, dtype=torch.long)
        return inputs

    def _transform_data(self, batch):
        # Preprocess the images
        images, masks = [], []
        for i, m in zip(batch["image"], batch["mask"]):
            img = np.asarray(i, dtype=np.uint8)
            mask = np.asarray(m, dtype=np.uint8)
            # Resize the images and masks
            resized_outputs = self.image_resizer(image=img, mask=mask)
            images.append(resized_outputs["image"])
            masks.append(resized_outputs["mask"])

        inputs = self.image_processor(
            images=images,
            return_tensors="pt",
        )
        inputs["labels"] = inputs["labels"] = torch.tensor(masks, dtype=torch.long)
        return inputs

    def setup(self, stage=None):
        dataset = self._create_dataset()
        train_ds = dataset["train"]
        valid_ds = dataset["validation"]
        test_ds = dataset["test"]

        if stage is None or stage == "fit":
            self.train_ds = train_ds.with_transform(self._transform_train_data)
            self.valid_ds = valid_ds.with_transform(self._transform_data)
        if stage is None or stage == "test" or stage == "predict":
            self.test_ds = test_ds.with_transform(self._transform_data)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)

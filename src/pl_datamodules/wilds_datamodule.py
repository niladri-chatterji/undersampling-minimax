import torch
import wilds
from wilds.common.grouper import CombinatorialGrouper

from omegaconf import DictConfig
from typing import List
from math import prod
from .base_datamodule import GroupDataModule
from torchvision.transforms import transforms
from ..datasets.wilds_dataset import WILDSDataset
from ..datasets.utils import ReweightedDataset


class WILDSDataModule(GroupDataModule):
    def __init__(
        self,
        dataset_name,
        resolution: List[int],
        train_transform: DictConfig,
        eval_transform: DictConfig,
        num_classes: int,
        groupby_fields: List[str],
        split_scheme="official",
        download=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.download = download
        self.split_scheme = split_scheme
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.dims = (3,) + tuple(resolution)
        self.num_classes = num_classes
        self.groupby_fields = groupby_fields

        if self.flatten_input:
            self.dims = (prod(self.dims),)

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""

        # Initializing wilds_dataset.WILDSDataset will create data dir and download
        # TODO: Check if actually downloads and unzip/tars
        _: wilds.datasets.wilds_dataset.WILDSDataset = wilds.get_dataset(
            dataset=self.dataset_name,
            root_dir=self.data_dir,
            download=self.download,
            split_scheme=self.split_scheme,
        )

    def setup(self, stage=None):
        """Load data. Set variables: self.train_dataset, self.data_val, self.test_dataset."""
        full_dataset = wilds.get_dataset(
            dataset=self.dataset_name,
            root_dir=self.data_dir,
            download=self.download,
            split_scheme=self.split_scheme,
        )
        train_transforms_list = initialize_transform(
            dataset=full_dataset, **self.train_transform
        )
        eval_transforms_list = initialize_transform(
            dataset=full_dataset, **self.eval_transform
        )

        if self.flatten_input:
            train_transforms_list.append(transforms.Lambda(lambda x: torch.flatten(x)))
            eval_transforms_list.append(transforms.Lambda(lambda x: torch.flatten(x)))
        self.train_transform = transforms.Compose(train_transforms_list)
        self.eval_transform = transforms.Compose(eval_transforms_list)

        grouper = CombinatorialGrouper(full_dataset, groupby_fields=self.groupby_fields)
        train_dataset = WILDSDataset(
            full_dataset.get_subset("train", transform=self.train_transform), grouper
        )
        # Note that some datasets from the WILDS dataset actually use other groupers for
        # eval, such as https://github.com/p-lambda/wilds/blob/e95bba8408aff524b48b96a4e7648df72773ad60/wilds/datasets/fmow_dataset.py#L203
        val_dataset = WILDSDataset(
            full_dataset.get_subset("val", transform=self.eval_transform), grouper
        )
        test_dataset = WILDSDataset(
            full_dataset.get_subset("test", transform=self.eval_transform), grouper
        )

        self.train_y_counter, self.train_g_counter, _ = self.compute_weights(
            train_dataset
        )
        print(f"Train class counts: {self.train_y_counter}")
        print(f"Train group counts: {self.train_g_counter}")
        self.val_y_counter, self.val_g_counter, val_weights = self.compute_weights(
            val_dataset
        )
        print(f"Val class counts: {self.val_y_counter}")
        print(f"Val group counts: {self.val_g_counter}")
        self.test_y_counter, self.test_g_counter, test_weights = self.compute_weights(
            test_dataset
        )
        print(f"Test class counts: {self.test_y_counter}")
        print(f"Test group counts: {self.test_g_counter}")

        val_dataset = ReweightedDataset(val_dataset, weights=val_weights)
        test_dataset = ReweightedDataset(test_dataset, weights=test_weights)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset


def initialize_transform(transform_name, config, dataset):
    if transform_name is None:
        return []
    elif transform_name == "bert":
        return initialize_bert_transform(config)
    elif transform_name == "image_base":
        return initialize_image_base_transform(config, dataset)
    elif transform_name == "image_resize_and_center_crop":
        return initialize_image_resize_and_center_crop_transform(config, dataset)
    elif transform_name == "poverty_train":
        return initialize_poverty_train_transform()
    else:
        raise ValueError(f"{transform_name} not recognized")


def initialize_bert_transform(config):
    # TODO: update this when we use BERT
    assert "bert" in config.model
    assert config.max_token_length is not None

    tokenizer = getBertTokenizer(config.model)

    def transform(text):
        tokens = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=config.max_token_length,
            return_tensors="pt",
        )
        if config.model == "bert-base-uncased":
            x = torch.stack(
                (
                    tokens["input_ids"],
                    tokens["attention_mask"],
                    tokens["token_type_ids"],
                ),
                dim=2,
            )
        elif config.model == "distilbert-base-uncased":
            x = torch.stack((tokens["input_ids"], tokens["attention_mask"]), dim=2)
        x = torch.squeeze(x, dim=0)  # First shape dim is always 1
        return x

    return transform


def getBertTokenizer(model):
    from transformers import BertTokenizerFast, DistilBertTokenizerFast

    if model == "bert-base-uncased":
        tokenizer = BertTokenizerFast.from_pretrained(model)
    elif model == "distilbert-base-uncased":
        tokenizer = DistilBertTokenizerFast.from_pretrained(model)
    else:
        raise ValueError(f"Model: {model} not recognized.")

    return tokenizer


def initialize_image_base_transform(config, dataset):
    transform_steps = []
    if dataset.original_resolution is not None and min(
        dataset.original_resolution
    ) != max(dataset.original_resolution):
        crop_size = min(dataset.original_resolution)
        transform_steps.append(transforms.CenterCrop(crop_size))
    if config.target_resolution is not None and config.dataset != "fmow":
        transform_steps.append(transforms.Resize(config.target_resolution))
    transform_steps += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    return transform_steps


def initialize_image_resize_and_center_crop_transform(config, dataset):
    """
    Resizes the image to a slightly larger square then crops the center.
    """
    assert dataset.original_resolution is not None
    assert config.resize_scale is not None
    scaled_resolution = tuple(
        int(res * config.resize_scale) for res in dataset.original_resolution
    )
    if config.target_resolution is not None:
        target_resolution = config.target_resolution
    else:
        target_resolution = dataset.original_resolution
    transforms = [
        transforms.Resize(scaled_resolution),
        transforms.CenterCrop(target_resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    return transforms


def initialize_poverty_train_transform():
    transforms_ls = [
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.1),
        transforms.ToTensor(),
    ]
    rgb_transform = transforms.Compose(transforms_ls)

    def transform_rgb(img):
        # bgr to rgb and back to bgr
        img[:3] = rgb_transform(img[:3][[2, 1, 0]])[[2, 1, 0]]
        return img

    transform = transforms.Lambda(lambda x: transform_rgb(x))
    return [transform]
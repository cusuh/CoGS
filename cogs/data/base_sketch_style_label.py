import bisect
import os

import albumentations
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset


class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


class ImagePaths(Dataset):
    def __init__(self, sketch_paths, cond_paths, image_paths, class_labels, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_sketch"] = sketch_paths
        self.labels["file_path_style"] = cond_paths
        self.labels["file_path_image"] = image_paths
        self.labels["class_label"] = class_labels
        self._length = len(sketch_paths)

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
            if not self.random_crop:
                self.cropper = albumentations.Resize(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.Resize(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()

        example["image"] = self.preprocess_image(self.labels["file_path_image"][i])
        example["sketch"] = self.preprocess_image(self.labels["file_path_sketch"][i])
        example["style"] = self.preprocess_image(self.labels["file_path_style"][i])
        example["label"] = self.labels["class_label"][i]
        for k in self.labels:
            example[k] = self.labels[k][i]

        return example

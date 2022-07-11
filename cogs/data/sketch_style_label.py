import os, tarfile, glob, shutil
import random
import yaml
import numpy as np
from tqdm import tqdm
from PIL import Image
import albumentations
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from cogs.data.base_sketch_style_label import ImagePaths
from cogs.util import download, retrieve
from collections import Counter


def give_synsets_from_indices(indices, path_to_yaml="data/imagenet_idx_to_synset.yaml"):
    synsets = []
    with open(path_to_yaml) as f:
        di2s = yaml.load(f)
    for idx in indices:
        synsets.append(str(di2s[idx]))
    print("Using {} different synsets for construction of Restriced Imagenet.".format(len(synsets)))
    return synsets


def str_to_indices(string):
    """Expects a string in the format '32-123, 256, 280-321'"""
    assert not string.endswith(","), "provided string '{}' ends with a comma, pls remove it".format(string)
    subs = string.split(",")
    indices = []
    for sub in subs:
        subsubs = sub.split("-")
        assert len(subsubs) > 0
        if len(subsubs) == 1:
            indices.append(int(subsubs[0]))
        else:
            rang = [j for j in range(int(subsubs[0]), int(subsubs[1]))]
            indices.extend(rang)
    return sorted(indices)


class SketchBase(Dataset):
    def __init__(self, config=None):
        self.config = config or OmegaConf.create()
        if not type(self.config)==dict:
            self.config = OmegaConf.to_container(self.config)
        self._prepare()
        self._prepare_synset_to_human()
        self._prepare_idx_to_synset()
        self._load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def _prepare(self):
        raise NotImplementedError()

    def _filter_relpaths(self, relpaths):

        if "sub_indices" in self.config:
            indices = str_to_indices(self.config["sub_indices"])
            synsets = give_synsets_from_indices(indices, path_to_yaml=self.idx2syn)  # returns a list of strings
            files = []
            for rpath in relpaths:
                syn = rpath.split("/")[0]
                if syn in synsets:
                    files.append(rpath)
            return files
        else:
            return relpaths

    def _prepare_synset_to_human(self):
        SIZE = 2655750
        URL = "https://heibox.uni-heidelberg.de/f/9f28e956cd304264bb82/?dl=1"
        self.human_dict = os.path.join(self.sketch_root, "synset_human.txt")
        if (not os.path.exists(self.human_dict) or
                not os.path.getsize(self.human_dict)==SIZE):
            download(URL, self.human_dict)

    def _prepare_idx_to_synset(self):
        URL = "https://heibox.uni-heidelberg.de/f/d835d5b6ceda4d3aa910/?dl=1"
        self.idx2syn = os.path.join(self.sketch_root, "index_synset.yaml")
        if (not os.path.exists(self.idx2syn)):
            download(URL, self.idx2syn)

    def _load(self):
        with open(self.txt_filelist, 'r') as f:
            data = f.read().splitlines()

        sketches = []
        styles = []
        gt_images = []
        for d in data:
            sketch, style, gt = d.split(',')
            sketches.append(sketch)
            styles.append(style)
            gt_images.append(gt)

        self.relpaths = sketches
        self.synsets = [p.split("/")[0] for p in sketches]
        self.abspaths_sketches = [os.path.join(self.sketch_root, p) for p in sketches]
        self.abspaths_styles = [os.path.join(self.style_root, p) for p in styles]
        self.abspaths_images = [os.path.join(self.image_root, p) for p in gt_images]

        unique_synsets = np.unique(self.synsets)
        class_dict = dict((synset, i) for i, synset in enumerate(unique_synsets))
        self.class_labels = [class_dict[s] for s in self.synsets]

        with open(self.human_dict, "r") as f:
            human_dict = f.read().splitlines()
            human_dict = dict(line.split(maxsplit=1) for line in human_dict)

        self.human_labels = [human_dict[s] for s in self.synsets]

        labels = {
            "relpath": np.array(self.relpaths),
            "synsets": np.array(self.synsets),
            "class_label": np.array(self.class_labels),
            "human_label": np.array(self.human_labels),
        }
        self.data = ImagePaths(self.abspaths_sketches,
                               self.abspaths_styles,
                               self.abspaths_images,
                               self.class_labels,
                               labels=labels,
                               size=retrieve(self.config, "size", default=0),
                               random_crop=self.random_crop)


class SketchTrain(SketchBase):
   NAME = "SketchTrain"

   def _prepare(self):
       self.random_crop = retrieve(self.config, "ImageNetTrain/random_crop",
                                   default=True)
       self.sketch_root = self.config["sketch_path"]
       self.image_root = self.config["image_path"]
       self.style_root = self.config["style_path"]
       self.txt_filelist = self.config['file_list']


class SketchValidation(SketchBase):
    NAME = "SketchVal"

    def _prepare(self):
        self.random_crop = retrieve(self.config, "ImageNetValidation/random_crop",
                                    default=False)
        self.sketch_root = self.config["sketch_path"]
        self.image_root = self.config["image_path"]
        self.style_root = self.config["style_path"]
        self.txt_filelist = self.config['file_list']

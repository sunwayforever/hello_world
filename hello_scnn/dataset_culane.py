#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2022-08-16
import json
import os

import pickle
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import util
import config
from collections import defaultdict

from data_augment import flip


class Culane(Dataset):
    def __init__(self, args, mode):
        super(Culane, self).__init__()
        self.args = args
        self.mode = mode
        self.load_data()

    def load_data(self):
        label_file = os.path.join("label", "culane_{}.dat".format(self.mode))
        with open(label_file, "rb") as f:
            self.label_data = pickle.load(f)

    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, idx):
        label_data = self.label_data[idx]
        image_file = os.path.join(
            config.LABEL_DATA_PATH_CULANE, label_data[0][1:]
        )  # "/data/datasets/CULane" + “image相对路径”
        label_file = os.path.join(
            config.LABEL_DATA_PATH_CULANE, label_data[1][1:]
        )  # "/data/datasets/CULane" + “label相对路径”
        exist = label_data[2]
        exist = torch.from_numpy(np.array(exist)).type(torch.float32)

        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = util.normalize(
            util.to_tensor(util.resize(img, (config.IMAGE_W, config.IMAGE_H))),
            config.MEAN,
            config.STD,
        )

        label = cv2.imread(label_file)
        label = label[:, :, 0]
        label = util.resize(label, (config.IMAGE_W, config.IMAGE_H))
        label = torch.from_numpy(label).type(torch.long)

        sample = {
            "img": img,
            "label": label,
            "exist": exist,
        }
        if self.mode == "test":
            return sample
        return (sample, flip(sample))


if __name__ == "__main__":
    data = Culane("train")
    print(data[1])

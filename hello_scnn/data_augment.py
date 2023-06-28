#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2022-08-16 14:01
import cv2
import numpy as np
import torch


def flip(sample):
    img = torch.flip(sample["img"], [2])
    exist = torch.flip(sample["exist"], [0])
    old_label = torch.flip(sample["label"], [1])
    label = torch.zeros_like(old_label)
    for i in range(5):
        label[np.where(old_label == i)] = (5 - i) % 5

    return {
        "img": img,
        "label": label,
        "exist": exist,
    }

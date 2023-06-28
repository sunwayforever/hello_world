#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2022-08-16 15:21
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_tusimple import Tusimple
from scnn_vgg import SCNNVgg
from scnn_mobilenet import SCNNMobileNet
from torchmetrics import F1Score
import sweep

device = torch.device("cuda:0")


def evaluate():
    print("evaluating")
    test_dataset = Tusimple(args, "test")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    net = None
    if args.model == "vgg":
        net = SCNNVgg(args, pretrained=True)
    if args.model == "mobilenet":
        net = SCNNMobileNet(args, pretrained=True)

    net = net.to(device)
    net.eval()
    save_dict = torch.load(net.get_model_name())
    net.load_state_dict(save_dict["net"])

    progress = tqdm(range(len(test_loader)))
    total_loss = 0.0
    f1score = F1Score(num_classes=5, average="none", mdmc_reduce="global").to(device)
    for _, sample in enumerate(test_loader):
        img = sample["img"].to(device)
        label = sample["label"].to(device)
        exist = sample["exist"].to(device)
        seg_pred, exist_pred, loss = net(img, label, exist)
        progress.set_description(f"loss: {loss.item():.3f}")
        progress.update(1)
        f1score.update(seg_pred, label)
        total_loss += loss.item()

    score = f1score.compute()
    progress.set_description(
        f"mean loss: {total_loss/len(test_loader):.3f},f1score: {score}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vgg", "mobilenet"], default="mobilenet")
    sweep.apply_model_config(parser)
    args = parser.parse_args()

    evaluate()

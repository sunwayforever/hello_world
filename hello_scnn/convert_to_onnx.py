#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2022-07-18 20:14
import torch
import torchvision
import argparse
import time

import numpy as np
from scnn_vgg import SCNNVgg
from scnn_mobilenet import SCNNMobileNet

import config
import sweep

def convert():
    net = None
    if args.model == "vgg":
        print("load vgg model")
        net = SCNNVgg(args, pretrained=True)
    if args.model == "mobilenet":
        print("load mobilenet model")
        net = SCNNMobileNet(args, pretrained=True)

    save_dict = torch.load(net.get_model_name())
    net.load_state_dict(save_dict["net"])
    net.eval()

    print("convert model to ONNX")
    dummy_input = torch.randn(1, 3, config.IMAGE_H, config.IMAGE_W)
    input_names = ["image"]
    output_names = ["seg_pred", "exist"]

    torch.onnx.export(
        net,
        dummy_input,
        "hello_scnn.onnx",
        opset_version=11,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=True,
    )
    print("Model has been converted to ONNX")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vgg", "mobilenet"], default="mobilenet")
    sweep.apply_model_config(parser)
    args = parser.parse_args()
    convert()

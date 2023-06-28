#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2022-08-18 21:26
def apply_training_config(parser):
    parser.add_argument("--dataset", choices=["culane", "tusimple"], default="tusimple")
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--max_learning_rate", type=float, default=0.167)
    parser.add_argument("--augment", choices=["true", "false"], default="true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", choices=["vgg", "mobilenet"], default="mobilenet")


def apply_model_config(parser):
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--n_message_passing", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--scale_background", type=float, default=0.4)
    parser.add_argument("--scale_seg", type=float, default=1.0)

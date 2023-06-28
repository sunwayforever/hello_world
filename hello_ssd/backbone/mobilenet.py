#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2021-04-13 15:04
import tensorflow as tf
from tensorflow.keras import Model, Sequential
import tensorflow.keras.layers as layers

from config import *
import util

FEATURE_MAP_SIZES = [19, 10, 5, 3, 2, 1]
N_ANCHORS = 2268

class MobileNet(Model):
    def __init__(self, weights):
        super().__init__()
        self.bn = util.L2Normalization()

        input = tf.keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3))
        mobile_net = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights=weights,
            input_tensor=input,
        )
        self.model = tf.keras.Model(
            inputs=input,
            outputs=[
                mobile_net.get_layer("out_relu").output,
                mobile_net.get_layer("block_13_expand_relu").output,
            ],
        )

    def call(self, x):
        x, feature1 = self.model(x)
        return x, (self.bn(feature1), x)

    # NOTE: conf layer 一共 6 层, 用来输入不同 level 的 anchor 对应的 conf,
    # 其中每层输出的 channel 前 4*xxx,6*xxx 和 box.py 中生成 anchor 的部分是对应的
    def get_conf_layers(self):
        return [
            # NOTE: 由于 stride 为 1, padding 为 same, 所以 conv2d 的输出尺寸与
            # 输入相同
            layers.SeparableConv2D(4 * N_CLASSES, kernel_size=3, padding="same"),
            layers.SeparableConv2D(6 * N_CLASSES, kernel_size=3, padding="same"),
            layers.SeparableConv2D(6 * N_CLASSES, kernel_size=3, padding="same"),
            layers.SeparableConv2D(6 * N_CLASSES, kernel_size=3, padding="same"),
            layers.SeparableConv2D(4 * N_CLASSES, kernel_size=3, padding="same"),
            layers.Conv2D(4 * N_CLASSES, kernel_size=1),
        ]

    # NOTE: loc layer 一共 6 层, 用来输入不同 level 的 anchor 对应的 loc
    def get_loc_layers(self):
        return [
            layers.SeparableConv2D(4 * 4, kernel_size=3, padding="same"),
            layers.SeparableConv2D(6 * 4, kernel_size=3, padding="same"),
            layers.SeparableConv2D(6 * 4, kernel_size=3, padding="same"),
            layers.SeparableConv2D(6 * 4, kernel_size=3, padding="same"),
            layers.SeparableConv2D(4 * 4, kernel_size=3, padding="same"),
            layers.Conv2D(4 * 4, kernel_size=1),
        ]

    # NOTE: extra_feature_layers 输出的 channel 数并不重要, 但它的 feature map
    # 的 shape 必须依次是 5, 3, 2, 1 (主要通过 strides=2 达到这个目的)
    def get_extra_feature_layers(self):
        return [
            Sequential(
                [
                    layers.Conv2D(256, 1, activation="relu"),
                    layers.Conv2D(512, 3, strides=2, padding="same", activation="relu"),
                ]
            ),
            Sequential(
                [
                    layers.Conv2D(128, 1, activation="relu"),
                    layers.Conv2D(256, 3, strides=2, padding="same", activation="relu"),
                ]
            ),
            Sequential(
                [
                    layers.Conv2D(128, 1, activation="relu"),
                    layers.Conv2D(256, 3, strides=2, padding="same", activation="relu"),
                ]
            ),
            Sequential(
                [
                    layers.Conv2D(128, 1, activation="relu"),
                    layers.Conv2D(256, 3, strides=3, padding="same", activation="relu"),
                ]
            ),
        ]

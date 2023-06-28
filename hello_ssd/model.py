#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2021-03-31 14:05
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, Sequential
from tensorflow.keras.applications import VGG16
import tensorflow.keras.layers as layers
from tensorflow.keras import backend as K

from backbone import *
from config import *

# NOTE: https://arxiv.org/abs/1512.02325


class SSDModel(Model):
    def __init__(self, resume=False):
        super().__init__()
        self.backbone = Backbone(None if resume else "imagenet")
        self.extra_feature_layers = self.backbone.get_extra_feature_layers()
        self.conf_layers = self.backbone.get_conf_layers()
        self.loc_layers = self.backbone.get_loc_layers()
        self.dropout = [layers.Dropout(0.2) for _ in range(6)]
        if resume:
            print(f"load weights from {MODEL_WEIGHTS}")
            self.load_weights(MODEL_WEIGHTS)

    def _reset_classifier(self):
        self.confs = []
        self.locs = []
        self.index = 0

    def _classify(self, x):
        if USE_DROPOUT:
            x = self.dropout[self.index](x)
        conf = self.conf_layers[self.index](x)
        conf = tf.reshape(conf, [conf.shape[0], -1, N_CLASSES])
        loc = self.loc_layers[self.index](x)
        loc = tf.reshape(loc, [loc.shape[0], -1, 4])

        self.confs.append(conf)
        self.locs.append(loc)
        self.index += 1

    def call(self, x, training=True):
        K.set_learning_phase(training)
        self._reset_classifier()
        # mobilenet
        # NOTE: x: [1, 10, 10, 1280],
        # features[0]: [1, 19, 19, 576],
        # features[1]: [1, 10, 10, 1280]

        # features 表示 6 个 level 的前两个 level (1444, 600) 对应的 anchor 的输
        # 入由 backbone 产生
        x, features = self.backbone(x)
        for f in features:
            self._classify(f)
        # 后 4 个 level (5, 3, 2, 1) 通过 extra_feature_layers 来产生
        for layer in self.extra_feature_layers:
            x = layer(x)
            # NOTE: 四个 extra_feature_layers 输出的 shape 依次为:
            # [1, 5, 5, 512]
            # [1, 3, 3, 256]
            # [1, 2, 2, 256]
            # [1, 1, 1, 256]
            self._classify(x)
        # NOTE: 最后 6 个 level 的 anchor 数据被 concat 在一起, 长度为 2268
        return tf.concat(self.confs, axis=1), tf.concat(self.locs, axis=1)


if __name__ == "__main__":
    import numpy as np

    ssd = SSDModel()
    input = np.random.normal(size=(1, 300, 300, 3))
    confs, locs = ssd(input)
    print(confs.shape)
    print(locs.shape)

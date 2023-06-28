#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2021-03-31 17:41
import tensorflow as tf
import numpy as np
from tensorflow.keras.losses import SparseCategoricalCrossentropy, Huber

from config import *


class SSDLoss(object):
    def __init__(self):
        self.temp_cross_entropy = SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        self.cross_entropy = SparseCategoricalCrossentropy(
            from_logits=True, reduction="sum"
        )
        self.smooth_l1 = Huber(reduction="sum")

    # NOTE: 由于 2268 个 anchor 中一般只有几个 anchor 的 conf 为非 0, 导致标签中
    # 有大量的负样本, 通过 hard_negative_mining, 只选择几个误差最大的负样本参与
    # loss 的计算, 以避免数据不平衡的问题
    def _hard_negative_mining(self, gt_confs, confs):
        # confs: (N, 2268, 21),
        # gt_confs: (N, 2268)
        temp_loss = self.temp_cross_entropy(gt_confs, confs)
        pos_idx = gt_confs > 0
        num_pos = tf.reduce_sum(tf.dtypes.cast(pos_idx, tf.int32), axis=1)
        num_neg = num_pos * HARD_NEGATIVE_RATIO

        rank = tf.argsort(temp_loss, axis=1, direction="DESCENDING")
        rank = tf.argsort(rank, axis=1)
        neg_idx = rank < tf.expand_dims(num_neg, 1)

        return pos_idx, tf.math.logical_or(pos_idx, neg_idx)

    def __call__(self, gt_confs, gt_locs, confs, locs):
        pos_index, all_index = self._hard_negative_mining(gt_confs, confs)
        # NOTE: 计算 conf_loss 使用了 all_index
        conf_loss = self.cross_entropy(gt_confs[all_index], confs[all_index])
        # NOTE: 计算 loc_loss 时只选择了 pos_index, 因为 compute_ground_truth 时
        # 负样本的 loc 数据是无效的
        loc_loss = self.smooth_l1(gt_locs[pos_index], locs[pos_index])
        num_pos = tf.reduce_sum(tf.dtypes.cast(pos_index, tf.float32))
        return conf_loss / num_pos, loc_loss / num_pos


if __name__ == "__main__":
    loss = SSDLoss()
    confs = np.random.normal(size=(32, 100, 21)).astype("float32")
    locs = np.random.normal(size=(32, 100, 4)).astype("float32")

    gt_confs = np.random.randint(0, 21, size=(32, 100))
    gt_locs = np.random.normal(size=(32, 100, 4)).astype("float32")

    print(
        loss(gt_confs, gt_locs, (10 * np.eye(21)).astype("float32")[gt_confs], gt_locs)
    )
    # print(loss(gt_confs, gt_locs, confs, locs))

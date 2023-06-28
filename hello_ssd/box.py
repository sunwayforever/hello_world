#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2021-03-30 20:55
from collections import namedtuple
import numpy as np
import math

from backbone import *
from config import *

_anchors = None


# NOTE: gen_anchors 是提前计算好的, 最终会生成 2268 个 anchor 的坐标.
def gen_anchors():
    global _anchors
    if _anchors is not None:
        return _anchors

    boxes = []
    for lvl in range(N_ANCHOR_LEVELS):
        # NOTE: FEATURE_MAP_SIZES = [19, 10, 5, 3, 2, 1] 表示网络一共会输出
        # N_ANCHOR_LEVELS (6) 个不同大小 (level) 的 feature map. 例如:
        #
        # mobilenet 的 `out_relu` 层的 feature map 大小为 [1, 19, 19, 576], 接一个
        # conv2d(4*4, kernel=3,strides=1,padding=same) 输出为 19*19*4*4, 可以看作
        # 是 19*19*4 个 box 的坐标
        #
        # mobilenet 的 `block_13_expand_relu` 层的 feature map 大小为 [1, 10, 10, 1280], 接一个
        # conv2d(6*4, kernel=3,strides=1,padding=same) 输出为 10*10*6*4, 可以看作
        # 是 10*10*6 个 box 的坐标
        #
        #
        # lvl_0 的 anchor 个数: 19*19*(2+2) = 1444
        # lvl_1 的 anchor 个数: 100*(2+4)=600
        # lvl_2 的 anchor 个数: 25*(2+4)=150
        # lvl_3 的 anchor 个数: 9*(2+4)=54
        # lvl_4 的 anchor 个数: 4*(2+2)=16
        # lvl_5 的 anchor 个数: 1*(2+2)=4
        #
        # 总共为 2268
        #
        # ANCHOR_SCALES 代表不同 lvl 的 box 的大小: lvl 越大 box 越大
        # ANCHOR_RATIOS 决定了 box 的长宽比, ratio 越大 box 越长(或越宽)
        #
        # 分成多少个 level, 每个 level 里 box 的间隔, 大小, 形状, 多少等都取决于要
        # 检测的物体的大小, 多少, 形状等.
        #
        # 例如, 若物体普遍比较大, 则 lvl_0 可能并不需要, 而其它 lvl 可以适当增加 box
        # 的大小和数目. 以 google mediapipe 的人脸检测模型为例, 它只有两个 lvl, 而
        # 且所有 box 的长宽均为 1
        for i in range(FEATURE_MAP_SIZES[lvl]):
            for j in range(FEATURE_MAP_SIZES[lvl]):
                cx = (j + 0.5) / FEATURE_MAP_SIZES[lvl]
                cy = (i + 0.5) / FEATURE_MAP_SIZES[lvl]
                # NOTE: 两个正方形的 anchor
                boxes.append([cx, cy, ANCHOR_SCALES[lvl], ANCHOR_SCALES[lvl]])
                boxes.append(
                    [
                        cx,
                        cy,
                        math.sqrt(ANCHOR_SCALES[lvl] * ANCHOR_SCALES[lvl + 1]),
                        math.sqrt(ANCHOR_SCALES[lvl] * ANCHOR_SCALES[lvl + 1]),
                    ]
                )

                # NOTE: n*2 个长方形的 anchor
                for ratio in ANCHOR_RATIOS[lvl]:
                    r = math.sqrt(ratio)
                    boxes.append(
                        [cx, cy, ANCHOR_SCALES[lvl] * r, ANCHOR_SCALES[lvl] / r]
                    )
                    boxes.append(
                        [cx, cy, ANCHOR_SCALES[lvl] / r, ANCHOR_SCALES[lvl] * r]
                    )

    boxes = np.array(boxes)
    _anchors = np.clip(boxes, 0.0, 1.0)
    return _anchors


def compute_area(top_left, bottom_right):
    rect = np.clip(bottom_right - top_left, a_min=0.0, a_max=None)
    return np.product(rect, -1)


# NOTE: 计算两个 box 的覆盖比例: Interceptio Over Union
def compute_iou(boxes_a, boxes_b):
    # m, 4 -> m, 1, 4
    # the expand dim is used for broadcast
    boxes_a = np.expand_dims(boxes_a, 1)
    # n, 4 -> 1, n, 4
    boxes_b = np.expand_dims(boxes_b, 0)
    # m,n,2
    top_left = np.maximum(boxes_a[:, :, :2], boxes_b[:, :, :2])
    bottom_right = np.minimum(boxes_a[:, :, 2:], boxes_b[:, :, 2:])

    overlap = compute_area(top_left, bottom_right)
    area_a = compute_area(boxes_a[:, :, :2], boxes_a[:, :, 2:])
    area_b = compute_area(boxes_b[:, :, :2], boxes_b[:, :, 2:])

    return overlap / (area_a + area_b - overlap)


def center_to_corner(boxes):
    return np.hstack([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2])


def corner_to_center(boxes):
    return np.hstack(
        [(boxes[..., :2] + boxes[..., 2:]) / 2, boxes[..., 2:] - boxes[..., :2]]
    )


def encode(anchors, locs):
    # NOTE: encode 需要把 box 的坐标转换为相对于 anchor 的坐标 (anchor_box)
    #
    #       +--------+
    #    +--+-----+  |
    #    |  |    b|  |
    #    |  +a----+--+
    #    +--------+
    # 正常的转换如下:
    # (locs[:, :2] - anchors[:, :2]) / anchors[:, 2:], locs[:, 2:] / anchors[:, 2:])
    #
    # 下面的 0.1, 0.2, log 是额外通过一些变换修改了值的范围, 便于更好的训练

    locs = corner_to_center(locs)
    locs = np.hstack(
        [
            (locs[:, :2] - anchors[:, :2]) / anchors[:, 2:] / 0.1,
            np.log(locs[:, 2:] / anchors[:, 2:]) / 0.2,
        ]
    )
    return locs


def decode(anchors, locs):
    # NOTE: decoder 与 encoder 相反, 把相对于 anchor 的坐标转换为相对于整个图片
    # 的坐标
    locs = np.hstack(
        [
            locs[:, :2] * 0.1 * anchors[:, 2:] + anchors[:, :2],
            np.exp(locs[:, 2:] * 0.2) * anchors[:, 2:],
        ],
    )

    locs = center_to_corner(locs)
    return locs


def compute_ground_truth(boxes, labels):
    # NOTE: boxes [2,4], 表示 box 的坐标 (x_min, y_min, x_max, y_max)
    # labels [2,], box 所属的类别 (bicycle, bird, boat, ...)
    #
    # NOTE: 假设测试图片中有两个 box
    #
    # 先定义几个名词:
    #
    # 1. anchor, 表示 gen_anchors 生成的 anchor, 每个 anchor 有它的坐标
    #
    # 2. anchor_box, 表示 anchor 对应的 box 相对于 anchor 中心的坐标
    #
    # 3. box, 表示标签中的 box
    #
    # 模型的标签和输出 (loc) 是 anchor_box:
    # 在 compute_ground_truth 时需要根据 (box,anchor) 获得 anchor_box 做为标签
    # 在 inference 时从模型输出得到 anchor_box, 然后根据 (anchor_box, anchor) 获得最终 box
    #
    # NOTE: anchors [2268, 4]
    #
    # 一共 2268 个 anchor, (center_x, center_y, width, height).
    #
    anchors = gen_anchors()
    #
    # NOTE: iou [2268, 2]
    #
    # 每个 [anchor, box] 都需要计算一个 iou 值, 例如 iou[2266]=[0.54030361,
    # 0.58199], 表示 anchors[2266] 和 boxes[0] 的 iou 为 0.54030361, 和 boxes[1]
    # 的 iou 为 0.58199
    iou = compute_iou(center_to_corner(anchors), boxes)
    #
    # NOTE: best_box [2268,]
    #
    # 每个 anchor 与哪个 box 重合更大
    #
    best_box_iou = np.max(iou, axis=1)
    best_box_index = np.argmax(iou, axis=1)
    #
    # NOTE: conf [2268,], anchor 对应的 box 的类别, 可能会有多个 anchor 负责同一
    # 个 box
    #
    confs = labels[best_box_index]
    confs[best_box_iou < IOU_THRESHOLD] = 0
    #
    # NOTE: locs [2268,], anchor 对应的 box 的坐标
    #
    locs = boxes[best_box_index]
    #
    # NOTE: encode 会把 anchor 对应的 box 的坐标转换为相对于 anchor 中心的坐标
    # (即 anchor_box)
    #
    locs = encode(anchors, locs)
    return confs, locs


Box = namedtuple("Box", ["label", "score", "box"])

# NOTE: Not Max Supression
#
# 预测的结果中每个 anchor 都会输出一个 (label, score, anchor_box), 这个结果会有许多重复:
# 多个 anchor 的 anchor_box 对应同一个 box
#
# NMS 的计算方法是:
# 1. 把所有 anchor 按 score 排序, 结果为集合 X
# 2. 输出 X 中 score 最大的 A, 并且 X.pop(A)
# 3. 计算 X 和 A 的 IOU, 所有超过 threshold(例如 0.5) 的 anchor (B) 被认为是与 A 重复, X.pop(B)
# 4. 重复 2
#
def nms(boxes):
    if len(boxes) <= 0:
        return np.array([])
    box = np.array([[*(d.box), d.score] for d in boxes])

    x1 = np.array(box[:, 0])
    y1 = np.array(box[:, 1])
    x2 = np.array(box[:, 2])
    y2 = np.array(box[:, 3])
    score = np.array(box[:, 4])

    area = np.multiply(x2 - x1 + 1, y2 - y1 + 1)
    I = np.array(score.argsort())
    pick = []
    while len(I) > 0:
        best = I[-1]
        pick.append(best)
        xx1 = np.maximum(x1[best], x1[I[0:-1]])
        yy1 = np.maximum(y1[best], y1[I[0:-1]])
        xx2 = np.minimum(x2[best], x2[I[0:-1]])
        yy2 = np.minimum(y2[best], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = w * h
        iou = intersection / (area[best] + area[I[0:-1]] - intersection)
        I = I[np.where(iou <= 0.5)[0]]
    return [boxes[i] for i in pick]


if __name__ == "__main__":
    import cv2

    _anchors = gen_anchors()
    # 004954
    boxes = np.array(
        [
            [0.05333333333333334, 0.306, 0.8466666666666667, 0.964],
        ]
    )
    # boxes = np.array(
    #     [
    #         [0.05333333333333334, 0.306, 0.8466666666666667, 0.964],
    #         [0.2311111111111111, 0.06, 0.9244444444444444, 0.872],
    #     ]
    # )
    # labels = np.array([17, 15])
    labels = np.array([17])
    confs, locs = compute_ground_truth(boxes, labels)

    width = height = 300
    image = np.zeros((width, height, 3))

    locs = decode(_anchors, locs)
    anchors = center_to_corner(_anchors)
    total = 0
    for i in range(len(locs)):
        if confs[i] == 0:
            continue
        total += 1
        # NOTE: 图片中只有一个 box:
        #
        # [0.05333333333333334, 0.306, 0.8466666666666667, 0.964]
        #
        # 但会有 6 个 anchor 对应这一个 box, 因为这 6 个 anchor 与 box 的 IOU 足
        # 够大. 每个 anchor 记录的 box 的坐标 (locs) 是不同的 (因为经过了
        # encode), 但经过 decode 后算出来的坐标都是相同的
        #
        x1, y1, x2, y2 = locs[i]
        print(x1, y1, x2, y2)
        xx1, yy1, xx2, yy2 = anchors[i]
        cv2.rectangle(
            image,
            (int(x1 * width), int(y1 * height)),
            (int(x2 * width), int(y2 * height)),
            (0, 0, 255),
            1,
            1,
        )
        cv2.putText(
            image,
            f"{total}",
            (int(xx1 * width) + 5, int(yy1 * height) + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            image,
            (int(xx1 * width), int(yy1 * height)),
            (int(xx2 * width), int(yy2 * height)),
            (0, 255, 0),
            1,
            1,
        )

    cv2.imshow("", image)
    cv2.waitKey(0)

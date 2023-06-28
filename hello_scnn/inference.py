#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2021-12-30 15:06
import cv2
import torch
import torchvision
import argparse
import time
from tqdm import tqdm

import numpy as np
import torch.nn.functional as F
import onnxruntime as onnx

from scnn_vgg import SCNNVgg
from scnn_mobilenet import SCNNMobileNet

import util
import config
import sweep

fps_counter = []
fps_counter_N = 5


class TorchDetector(object):
    def __init__(self, args):
        if args.model == "vgg":
            self.net = SCNNVgg(args, pretrained=True)
        if args.model == "mobilenet":
            self.net = SCNNMobileNet(args, pretrained=True)
        save_dict = torch.load(self.net.get_model_name())
        self.net.load_state_dict(save_dict["net"])
        self.net.eval()

    def invoke(self, image):
        seg_pred, exist_pred, _ = self.net(image)
        seg_pred = seg_pred.detach().cpu().numpy()
        exist_pred = exist_pred.detach().cpu().numpy()
        return seg_pred[0], exist_pred


class OnnxDetector(object):
    def __init__(self, model):
        self.onnx_session = onnx.InferenceSession(model)

    def invoke(self, image):
        seg_pred, exist_pred = self.onnx_session.run(
            ["seg_pred", "exist"],
            {"image": image.numpy()},
        )
        return seg_pred[0], exist_pred


def inference_image(args, detector, image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img = util.resize(img, (config.IMAGE_W, config.IMAGE_H))
    data = util.normalize(util.to_tensor(img), config.MEAN, config.STD)
    data.unsqueeze_(0)

    seg_pred, exist_pred = detector.invoke(data)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    lane_img = np.zeros_like(img)

    color = np.array(
        [[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], dtype="uint8"
    )
    coord_mask = np.argmax(seg_pred, axis=0)
    prob = np.max(seg_pred, axis=0)

    if args.vis == "raw":
        for i in range(1, 5):
            x, y = np.where(np.logical_and(coord_mask == i, prob > config.MIN_PROB))
            lane_img[(x, y)] = color[i - 1]

    if args.vis == "line":
        for i in range(1, 5):
            x, y = np.where(np.logical_and(coord_mask == i, prob > config.MIN_PROB))
            if len(x) <= 10:
                continue
            p = np.polyfit(x, y, deg=2)
            y = np.poly1d(p)(x).astype(np.int)
            y = np.minimum(y, config.IMAGE_W - 1)
            lane_img[(x, y)] = color[i - 1]

    if args.vis == "fill":
        left_y, left_x = np.where(
            np.logical_and(coord_mask == 2, prob > config.MIN_PROB)
        )
        # left_y, left_x = np.where(coord_mask == 2)
        right_y, right_x = np.where(
            np.logical_and(coord_mask == 3, prob > config.MIN_PROB)
        )
        # right_y, right_x = np.where(coord_mask == 3)
        if len(left_x) >= 10 and len(right_x) >= 10:
            left_poly = np.polyfit(left_y, left_x, deg=1)
            right_poly = np.polyfit(right_y, right_x, deg=1)
            y = np.linspace(
                np.min(np.concatenate([left_y, right_y])), config.IMAGE_H
            ).astype(np.int32)
            left_x = np.poly1d(left_poly)(y).astype(np.int32)
            right_x = np.poly1d(right_poly)(y).astype(np.int32)

            left_points = np.array([[[xi, yi]] for xi, yi in zip(left_x, y)])
            right_points = np.array([[[xi, yi]] for xi, yi in zip(right_x, y)])
            points = np.concatenate((left_points, np.flip(right_points, 0)))
            cv2.fillPoly(lane_img, [points], color=[0, 255, 0])

    img = cv2.addWeighted(src1=lane_img, alpha=0.5, src2=img, beta=1.0, gamma=0.0)
    fps_counter.append(int(time.time() * 1000))
    if len(fps_counter) >= fps_counter_N:
        fps = fps_counter_N * 1000 // (fps_counter[-1] - fps_counter[0])
        fps_counter.pop(0)
        cv2.putText(
            img,
            f"FPS:{fps}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
    if args.output == "vis":
        cv2.imshow("", img)
        cv2.waitKey(1)
    return img


def inference(args):
    print(f"using {args.model}")
    detector = None
    if args.model == "vgg" or args.model == "mobilenet":
        detector = TorchDetector(args)
    if args.model == "onnx":
        detector = OnnxDetector("hello_scnn.onnx")

    if args.video != None:
        if args.output == "dump":
            out = cv2.VideoWriter(
                "dump.mp4",
                cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                10,
                (config.IMAGE_W, config.IMAGE_H),
            )
        cap = cv2.VideoCapture(args.video)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(range(length))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            img = inference_image(args, detector, frame)
            if args.output == "dump":
                pbar.update(1)
                out.write(img)
        pbar.close()
        if args.output == "dump":
            print("video dumped to dump.mp4")
    else:
        image = cv2.imread(args.image)
        img = inference_image(args, detector, image)
        if args.output == "dump":
            cv2.imwrite(f"dump.jpg", img)
            print("image dumped to dump.jpg")

    if args.output == "vis":
        while True:
            k = cv2.waitKey(0) & 0xFF
            if k == ord("q"):
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", choices=["dump", "vis"], required=True)
    parser.add_argument(
        "--model", choices=["vgg", "mobilenet", "onnx"], default="mobilenet"
    )
    parser.add_argument("--vis", choices=["raw", "line", "fill"], default="fill")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=str)
    source.add_argument("--video", type=str)
    sweep.apply_model_config(parser)
    args = parser.parse_args()
    inference(args)

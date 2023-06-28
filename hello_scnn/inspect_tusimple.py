#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2021-12-30 13:14
import json
import cv2

meta_data = json.loads(open("hello_tusimple.json").readline())

image_name = meta_data["raw_file"]
image = cv2.imread(image_name)

lane_y_points = meta_data["h_samples"]

for lane_index, lane_x_points in enumerate(meta_data["lanes"]):
    lane_points = [(x, y) for (x, y) in zip(lane_x_points, lane_y_points) if x > 0]
    for point_a, point_b in zip(lane_points[:-1], lane_points[1:]):
        cv2.line(image, tuple(point_a), tuple(point_b), (0, 255, 0), 2)

        for point in lane_points:
            cv2.circle(image, tuple(point), 2, (0, 0, 255), -1)

cv2.imshow("", image)
cv2.waitKey()

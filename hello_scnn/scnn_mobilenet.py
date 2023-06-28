import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


import config
from scnn import SCNN


class SCNNMobileNet(SCNN):
    def get_model_name(self):
        return "hello_scnn_mobilenet.pth"

    def net_init(self, K):
        input_w, input_h = config.IMAGE_W, config.IMAGE_H
        self.fc_input_feature = 5 * int(input_w / 16) * int(input_h / 16)
        self.backbone = models.mobilenet_v2(pretrained=self.pretrained).features[:7]
        # ----------------- SCNN part -----------------
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 8, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),  # (nB, 128, 36, 100)
        )

        # ----------------- add message passing -----------------
        self.message_passing = nn.ModuleList()
        self.message_passing.add_module(
            "up_down",
            nn.Conv2d(8, 8, (1, K), padding=(0, K // 2), bias=False),
        )
        self.message_passing.add_module(
            "down_up",
            nn.Conv2d(8, 8, (1, K), padding=(0, K // 2), bias=False),
        )
        self.message_passing.add_module(
            "left_right",
            nn.Conv2d(8, 8, (K, 1), padding=(K // 2, 0), bias=False),
        )
        self.message_passing.add_module(
            "right_left",
            nn.Conv2d(8, 8, (K, 1), padding=(K // 2, 0), bias=False),
        )
        # (nB, 128, 36, 100)

        # ----------------- SCNN part -----------------
        self.layer2 = nn.Sequential(
            nn.Dropout2d(self.args.dropout), nn.Conv2d(8, 5, 1)  # get (nB, 5, 36, 100)
        )

        self.layer3 = nn.Sequential(
            nn.Softmax(dim=1),  # (nB, 5, 36, 100)
            nn.AvgPool2d(2, 2),  # (nB, 5, 18, 50)
        )
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_feature, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.Sigmoid(),
        )

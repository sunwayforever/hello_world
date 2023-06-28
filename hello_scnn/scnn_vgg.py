import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# NOTE: arxiv
#
# scnn: 2017/12
# https://arxiv.org/pdf/1712.06080.pdf
#
# dilated_net: 2017/5
# https://towardsdatascience.com/review-dilated-convolution-semantic-segmentation-9d5a5bd768f5
# https://arxiv.org/pdf/1606.00915
#
# fcn for semantic segmentation: 2015/3
# https://towardsdatascience.com/review-fcn-semantic-segmentation-eb8c9b50d2d1
# https://arxiv.org/pdf/1411.4038.pdf
#

import config
from scnn import SCNN


class SCNNVgg(SCNN):
    def get_model_name(self):
        return "hello_scnn_vgg.pth"

    def net_init(self, ms_ks):
        input_w, input_h = config.IMAGE_W, config.IMAGE_H
        self.fc_input_feature = 5 * int(input_w / 16) * int(input_h / 16)
        # NOTE: backbone 用的是 vgg16 的特征提取部分, 但删除了 33,43 两层
        # maxpooling, 因为 pooling 会导致的空间信息丢失; 同时替换最后几层 conv2d
        # 为 dilation conv2d, 因为删除 pooling 会导致 receptive field 变小, 用
        # dilation conv2d 可以改善这种情况.
        #
        # 关于 pooling 与 semantic segmentation 参考 fcn
        #
        # 关于 dilation conv 与 semantic segmentation 参考 dilated_net
        #
        self.backbone = models.vgg16_bn(pretrained=self.pretrained).features
        # ----------------- process backbone -----------------
        for i in [34, 37, 40]:
            conv = self.backbone._modules[str(i)]
            dilated_conv = nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                conv.kernel_size,
                stride=conv.stride,
                padding=tuple(p * 2 for p in conv.padding),
                dilation=2,
                bias=(conv.bias is not None),
            )
            dilated_conv.load_state_dict(conv.state_dict())
            self.backbone._modules[str(i)] = dilated_conv
        self.backbone._modules.pop("33")
        self.backbone._modules.pop("43")

        # ----------------- SCNN part -----------------
        self.layer1 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),  # (nB, 128, 36, 100)
        )

        # ----------------- add message passing -----------------
        self.message_passing = nn.ModuleList()
        self.message_passing.add_module(
            "up_down",
            nn.Conv2d(128, 128, (1, ms_ks), padding=(0, ms_ks // 2), bias=False),
        )
        self.message_passing.add_module(
            "down_up",
            nn.Conv2d(128, 128, (1, ms_ks), padding=(0, ms_ks // 2), bias=False),
        )
        self.message_passing.add_module(
            "left_right",
            nn.Conv2d(128, 128, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False),
        )
        self.message_passing.add_module(
            "right_left",
            nn.Conv2d(128, 128, (ms_ks, 1), padding=(ms_ks // 2, 0), bias=False),
        )
        # (nB, 128, 36, 100)

        # ----------------- SCNN part -----------------
        self.layer2 = nn.Sequential(
            nn.Dropout2d(0.1), nn.Conv2d(128, 5, 1)  # get (nB, 5, 36, 100)
        )

        self.layer3 = nn.Sequential(
            nn.Softmax(dim=1),  # (nB, 5, 36, 100)
            nn.AvgPool2d(2, 2),  # (nB, 5, 18, 50)
        )
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_feature, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid(),
        )

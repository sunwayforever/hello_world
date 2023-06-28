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


class SCNN(nn.Module):
    # NOTE: K 是 message_passing 的 kernel 的大小, 必须为奇数, 才能保证 conv 后
    # 尺寸不变
    def __init__(self, args, pretrained=True):
        super(SCNN, self).__init__()
        self.pretrained = pretrained
        self.args = args
        self.net_init(args.k)

        self.scale_background = args.scale_background
        self.scale_seg = args.scale_seg
        self.scale_exist = 0.1

        self.ce_loss = nn.CrossEntropyLoss(
            weight=torch.tensor([self.scale_background, 1, 1, 1, 1])
        )
        self.bce_loss = nn.BCELoss()

    def forward(self, img, seg_gt=None, exist_gt=None):
        # img: [1, 3, 288, 512]
        x = self.backbone(img)
        # x: [1, 512, 36, 64]
        x = self.layer1(x)
        # x: [1, 128, 36, 64]
        x = self.message_passing_forward(x)
        # x: [1, 128, 36, 64]
        x = self.layer2(x)
        #
        # x: [1, 5, 36, 64]
        seg_pred = F.interpolate(x, scale_factor=8, mode="bilinear", align_corners=True)
        # seg_pred: [1, 5, 288, 512]
        x = self.layer3(x)
        # x: [1, 5, 18, 32]
        x = x.view(-1, self.fc_input_feature)
        exist_pred = self.fc(x)
        # exist_pred: [1, 4]

        loss = None
        if seg_gt is not None:
            # training
            loss_seg = self.ce_loss(seg_pred, seg_gt)
            loss_exist = self.bce_loss(exist_pred, exist_gt)
            loss = loss_seg * self.scale_seg + loss_exist * self.scale_exist

        return seg_pred, exist_pred, loss

    def message_passing_forward(self, x):
        # NOTE: message_passing_forward 是 SCNN 最核心的部分
        # vertical 和 reverse zip 起来, 一共需要做 4 遍 message_passing:
        # up, down, left, right
        # see https://arxiv.org/pdf/1712.06080.pdf page 3
        Vertical = [True, True, False, False]
        Reverse = [False, True, False, True]
        for ms_conv, v, r in zip(
            self.message_passing[: self.args.n_message_passing],
            Vertical[: self.args.n_message_passing],
            Reverse[: self.args.n_message_passing],
        ):
            x = self.message_passing_once(x, ms_conv, v, r)
        return x

    def message_passing_once(self, x, conv, vertical=True, reverse=False):
        nB, C, H, W = x.shape
        # NOTE: down 时的计算步骤:
        # 1. 按 H 方向切成 H 个 (C,1,W) 的 slice
        # 2. out[0]=slice[0]
        # 3. out[1]=slice[1]+conv(out[0])
        # 4. out[2]=slice[2]+conv(out[1])
        # ...
        #
        # 计算完以后 out shape 和 input shape (基本) 是一样的, 因为 stride = 1,
        # padding=k//2, 而 O=(w+2p-k)/s+1.
        #
        # 另外整个过程和 rnn 很像: slice 是 input 序列, out 是 hidden state 序列
        #
        # 按论文的说法, 这样操作更容易捕获长条形状的物体 (比如车道线) 或大的物体
        # 的空间信息
        #
        # Intuition:
        # 为了让 cnn 能捕获大物体的信息, 需要有较大的 receptive field:
        #
        # 1. 通过 pooling
        # 2. 通过更大的 kernel
        #
        # 前者会导致信息丢失
        # 后者计算量太大
        #
        # SCNN 的作法是通过类似 rnn 的作法让空间中不同位置的信息能传递(合并)在一
        # 起, 让网络能看到整个空间的信息
        #
        if vertical:
            slices = [x[:, :, i : (i + 1), :] for i in range(H)]
            dim = 2
        else:
            slices = [x[:, :, :, i : (i + 1)] for i in range(W)]
            dim = 3
        if reverse:
            slices = slices[::-1]

        out = [slices[0]]
        for i in range(1, len(slices)):
            out.append(slices[i] + F.relu(conv(out[i - 1])))
        if reverse:
            out = out[::-1]
        return torch.cat(out, dim=dim)

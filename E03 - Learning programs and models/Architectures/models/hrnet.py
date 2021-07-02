import torch
import torch.nn as nn
import torch.nn.functional as F

from .segbase import SegBaseModel

__all__ = ['HRNet']


class HRNet(SegBaseModel):
    def __init__(self, nclass, backbone_name='hrnet_w30', norm_layer=nn.BatchNorm2d, BN_MOMENTUM = 0.01, FINAL_CONV_KERNEL = 1):
        self.backbone_name = backbone_name
        self.nclass = nclass
        self.norm_layer = norm_layer
        super(HRNet, self).__init__(backbone_name=self.backbone_name, nclass=self.nclass, need_backbone=True)

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels=self.backbone.last_inp_channels,
                out_channels=self.backbone.last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            norm_layer(self.backbone.last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=self.backbone.last_inp_channels,
                out_channels=nclass,
                kernel_size=FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if FINAL_CONV_KERNEL == 3 else 0)
        )

    def forward(self, x):
        height, width = x.shape[2], x.shape[3]
        x = self.backbone(x)

        x = self.head(x)
        x = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=True)

        return x
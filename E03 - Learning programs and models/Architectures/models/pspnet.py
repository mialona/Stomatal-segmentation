"""Pyramid Scene Parsing Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .segbase import SegBaseModel
from ..modules import _FCNHead, PyramidPooling

__all__ = ['PSPNet']


class PSPNet(SegBaseModel):
    r"""Pyramid Scene Parsing Network
    Reference:
        Zhao, Hengshuang, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia.
        "Pyramid scene parsing network." *CVPR*, 2017
    """

    def __init__(self, nclass, backbone_name="resnet101"):
        self.backbone_name = backbone_name
        self.nclass = nclass
        super(PSPNet, self).__init__(backbone_name=self.backbone_name, nclass=self.nclass, need_backbone=True)

        self.head = _PSPHead(self.nclass)

        self.__setattr__('decoder', ['head'])

    def forward(self, x):
        size = x.size()[2:]
        _, _, c3, c4 = self.backbone(x)

        x = self.head(c4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x


class _PSPHead(nn.Module):
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_PSPHead, self).__init__()
        self.psp = PyramidPooling(2048, norm_layer=norm_layer)
        self.block = nn.Sequential(
            nn.Conv2d(4096, 512, 3, padding=1, bias=False),
            norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(512, nclass, 1)
        )

    def forward(self, x):
        x = self.psp(x)
        return self.block(x)


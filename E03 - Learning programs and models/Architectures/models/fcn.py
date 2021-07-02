from __future__ import division

import torch.nn.functional as F

from .segbase import SegBaseModel
from ..modules import _FCNHead

__all__ = ['FCN']


class FCN(SegBaseModel):
    def __init__(self, nclass, backbone_name="resnet101"):
        self.backbone_name = backbone_name
        self.nclass = nclass
        super(FCN, self).__init__(backbone_name=self.backbone_name, nclass=self.nclass, need_backbone=True)

        self.head = _FCNHead(2048, self.nclass)
        self.__setattr__('decoder', ['head'])

    def forward(self, x):
        size = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)

        x = self.head(c4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x

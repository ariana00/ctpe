import math
import torch
from torch import nn
from torch.nn import functional as NF
from torchvision.models import resnet50

import os, sys
CURR_DIR = os.path.dirname(__file__)
if __name__ == "__main__": sys.path.insert(0, os.path.join(CURR_DIR, ".."))
import myutils

from .penet import PENet


class MIL_SoftmaxAttention(nn.Module):
    def __init__(self, inplanes, midplanes, dropout=0.1):
        super(MIL_SoftmaxAttention, self).__init__()
        # input is BxN
        self.inplanes = inplanes
        self.proj = nn.Sequential(
                    nn.Linear(inplanes, inplanes),
                    nn.LeakyReLU(inplace=False),
                    nn.Linear(inplanes, 1)
        )
        self.softmax = nn.Softmax(dim=1)
        self.classifier = nn.Sequential(
                    nn.Linear(inplanes, midplanes),
                    nn.LeakyReLU(inplace=False),
                    # nn.Dropout(dropout),
                    nn.Linear(midplanes, 1),
                    nn.Sigmoid()
        )

    def forward(self, bag_feature, batch_size):
        #import pdb
        #pdb.set_trace()
        weight = self.proj(bag_feature).view(batch_size, -1, 1)
        weight = self.softmax(weight)
        bag_feature = bag_feature.view(batch_size, -1, self.inplanes)
        # print (weight)
        agg_feat = torch.sum(weight * bag_feature, dim=1, keepdim=False)   # inplanes
        y_pred = self.classifier(agg_feat)
        y_pred = torch.clamp(y_pred, min=1e-5, max=1. - 1e-5)
        y_pred = y_pred.view(batch_size)
        return y_pred



class PE_MIL(nn.Module):
    def __init__(self, penet_params, freeze_penet=True, device=0,):
        super(PE_MIL, self).__init__()
        self.penet = PENet(**penet_params)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d((None, 1, 1))
        self.mil_model = MIL_SoftmaxAttention(inplanes=768, midplanes=512, dropout=0.1)
        if freeze_penet:               
            for param in self.penet.parameters():
                #print(param)
                param.requires_grad = False

        self.fine_tuning_param('out_conv')
        self.fine_tuning_param('asp_pool')
        self.fine_tuning_param('encoders.2')
        self.fine_tuning_param('encoders.1')
        self.fine_tuning_param('encoders.0')
        self.fine_tuning_param('in_conv')

        # freeze bn
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                for p in m.parameters():
                    p.requires_grad = False

        # debug
        # self.feat_1 = torch.ones((16, 256, 3, 12, 12)).cuda()
        # self.feat_0 = torch.zeros((16, 256, 3, 12, 12)).cuda()

    # input is BxCx24xHxW, penet output is Nx256x3x12x12
    def forward(self, x, batch_size):
        b = x.size(0)
        feat = self.penet.extract_feat_batchly(x)

        bag_feature = self.max_pool(feat).view(b, -1)
        # print (label, bag_feature)
        y_pred = self.mil_model(bag_feature, batch_size)

        return y_pred

    def fine_tuning_param(self, boundary_layer_name):
        for name, param in self.penet.named_parameters():
            if name.startswith(boundary_layer_name):
                #print(name)
                param.requires_grad = True
        #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.penet.parameters()), lr = 1e-4)
        return 
    # input is BxCx24xHxW, penet output is Nx256x3x12x12
    def forward_debug(self, x, label=None):
        b = x.size(0)

        # feat = self.penet.extract_feat_batchly(x)
        if label > 0.5:
            # feat[:] = 1
            feat = self.feat_1
        else:
            feat = self.feat_0
        b=16

        # import pdb
        # pdb.set_trace()
        bag_feature = self.max_pool(feat).view(b, -1)
        # print (label, bag_feature)
        y_pred = self.mil_model(bag_feature)
        return y_pred










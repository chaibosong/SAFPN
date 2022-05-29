# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn
from .attention_gate import AttentionGate
from .bottom_up import BottomUp
from ..anet.FPANet import FPNCNet
from ..anet.sca_attention2 import ExternaAtt
from ..anet import sca_attention2, sca_attention
from ..anet.BiFPN import BiFPN
from ..anet.efficientnet.model import BiFPN1


class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(
            self, in_channels_list, out_channels, conv_block, top_blocks=None
    ):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
      

        super(FPN, self).__init__()

        self.inner_blocks = []
        self.layer_blocks = []

        for idx, in_channels in enumerate(in_channels_list, 1):  
           
            inner_block = "fpn_inner{}".format(idx)

          
            layer_block = "fpn_layer{}".format(idx)

        
            inner_block_module = conv_block(in_channels, out_channels, 1)

           
            layer_block_module = conv_block(out_channels, out_channels, 3, 1)

 
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)

            
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)

      
        self.top_blocks = top_blocks
        # self.bifpn = nn.Sequential(
        #     *[BiFPN(256,
        #             in_channels_list,
        #             True if _ == 0 else False,
        #             attention=True,
        #             )
        #       for _ in range(2)])
        self.bifpn = nn.Sequential(
            *[BiFPN(256,
                    in_channels_list,
                    True if _ == 0 else False,
                    attention=True ,
                    )
              for _ in range(2)])

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """

        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        # print('-----------count----------', count)
        # print('layer_last shape: ', last_inner.shape)
        results = []

      
        c5 = getattr(self, self.layer_blocks[-1])(last_inner)
        results.append(c5)
        cx_results = []
        cx_results.append(c5)

        results=self.bifpn(x)
        
        if self.top_blocks is not None:
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)

        return tuple(results)



class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]

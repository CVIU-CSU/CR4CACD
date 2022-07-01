import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from .convfc_bbox_head import ConvFCBBoxHead
import math


@HEADS.register_module()
class RoIAttentionConvFCBBoxHead(ConvFCBBoxHead):

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 attention_hidden_channels=128,
                 attention_pool_size=2,
                 with_golbal=True,
                 *args,
                 **kwargs):
        super().__init__(num_shared_convs, num_shared_fcs, num_cls_convs, num_cls_fcs, num_reg_convs, num_reg_fcs,
                         conv_out_channels, fc_out_channels, conv_cfg, norm_cfg, *args, **kwargs)
        self.attention_hidden_channels = attention_hidden_channels
        self.conv_out_channels = conv_out_channels
        #self.ds_conv = nn.Conv2d(conv_out_channels, conv_out_channels, attention_pool_size, stride=attention_pool_size)
        self.q_conv = nn.Conv2d(conv_out_channels, attention_hidden_channels, 1)
        
        #self.k_conv = nn.Conv2d(conv_out_channels, attention_hidden_channels, 1)
        #self.v_conv = nn.Conv2d(conv_out_channels, attention_hidden_channels, 1)
        self.k_conv = nn.Conv2d(conv_out_channels, attention_hidden_channels, 1, stride=attention_pool_size)
        self.v_conv = nn.Conv2d(conv_out_channels, attention_hidden_channels, 1, stride=attention_pool_size)
        self.y_conv = nn.Conv2d(attention_hidden_channels, conv_out_channels, 1)
        self.attention_pool_size = attention_pool_size
        self.with_golbal = with_golbal
        if self.with_golbal:
            self.q_conv_g = nn.Conv2d(conv_out_channels, attention_hidden_channels, 1)
            #self.ds_conv_g = nn.Conv2d(conv_out_channels, conv_out_channels, attention_pool_size, stride=attention_pool_size)
            #self.k_conv_g = nn.Conv2d(conv_out_channels, attention_hidden_channels, 1)
            #self.v_conv_g = nn.Conv2d(conv_out_channels, attention_hidden_channels, 1)
            self.k_conv_g = nn.Conv2d(conv_out_channels, attention_hidden_channels, 1, stride=attention_pool_size)
            self.v_conv_g = nn.Conv2d(conv_out_channels, attention_hidden_channels, 1, stride=attention_pool_size)
            self.y_conv_g = nn.Conv2d(attention_hidden_channels, conv_out_channels, 1)
            
        
    def init_weights(self):
        super(RoIAttentionConvFCBBoxHead, self).init_weights()
        # conv layers are already initialized by ConvModule
        
    def attention(self, feature, roi_feats):
        BS, C, H, W = feature.shape
        BS_num_rois, C_roi, roi_h, roi_w = roi_feats.shape
        num_rois = BS_num_rois // BS
        Q = self.q_conv_g(roi_feats)  # (BS*num_rois, attention_hidden_channels, H, W)
        # stride=2
        #_x = F.max_pool2d(feature, self.attention_pool_size, self.attention_pool_size)
        #_x = self.ds_conv_g(feature)
        #_H, _W = H // self.attention_pool_size, W // self.attention_pool_size
        
        _H, _W = math.ceil(H / self.attention_pool_size), math.ceil(W / self.attention_pool_size)
        K = self.k_conv_g(feature)  # (BS*num_rois, attention_hidden_channels, _H, _W)
        V = self.v_conv_g(feature)  # (BS*num_rois, attention_hidden_channels, _H, _W)

        Q = Q.permute(0, 2, 3, 1)  # (BS*num_rois, H, W, attention_hidden_channels)
        Q = Q.reshape(BS, num_rois, roi_h, roi_w, self.attention_hidden_channels)
        Q = Q.reshape(BS, num_rois*roi_h*roi_w, self.attention_hidden_channels)  # (BS, num_rois*H*W, attention_hidden_channels)

        K = K.permute(0, 2, 3, 1)  # (BS*num_rois, _H, _W, attention_hidden_channels)
        K = K.reshape(BS, _H*_W, self.attention_hidden_channels)  # (BS, num_rois*_H*_W, attention_hidden_channels)

        V = V.permute(0, 2, 3, 1)  # (BS*num_rois, _H, _W, attention_hidden_channels)
        V = V.reshape(BS, _H*_W, self.attention_hidden_channels)  # (BS, num_rois*_H*_W, attention_hidden_channels)

        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        WEIGHTS = torch.bmm(Q, K.permute(0, 2, 1))  # (BS, num_rois*H*W, _H*_W)
        WEIGHTS = torch.softmax(WEIGHTS, dim=2)

        Y = torch.bmm(WEIGHTS, V)  # (BS, num_rois*H*W, attention_hidden_channels)
        Y = Y.reshape(BS, num_rois, roi_h, roi_w, self.attention_hidden_channels)
        Y = Y.reshape(BS*num_rois, roi_h, roi_w, self.attention_hidden_channels)
        Y = Y.permute(0, 3, 1, 2)
        Y = Y.contiguous()
        y = self.y_conv_g(Y)
        y = y.contiguous()

        roi_enhanced = roi_feats + y
        return roi_enhanced
        #return y
        
    def forward(self, x, feats):
        """

        :param x: shape (BS, num_rois, C, H, W)
        :return:
        """
        BS, num_rois, C, H, W = x.shape
        x = x.reshape(BS*num_rois, C, H, W)

        Q = self.q_conv(x)  # (BS*num_rois, attention_hidden_channels, H, W)
        _H, _W = math.ceil(H / self.attention_pool_size), math.ceil(W / self.attention_pool_size)
        #_H, _W = H // self.attention_pool_size, W // self.attention_pool_size
        #_x = F.max_pool2d(x, self.attention_pool_size, self.attention_pool_size)
        #_x = self.ds_conv(x)
        #layer_norm = nn.LayerNorm([self.conv_out_channels, _H, _W])
        #_x = layer_norm(_x)
        K = self.k_conv(x)  # (BS*num_rois, attention_hidden_channels, _H, _W)
        V = self.v_conv(x)  # (BS*num_rois, attention_hidden_channels, _H, _W)

        Q = Q.permute(0, 2, 3, 1)  # (BS*num_rois, H, W, attention_hidden_channels)
        Q = Q.reshape(BS, num_rois, H, W, self.attention_hidden_channels)
        Q = Q.reshape(BS, num_rois*H*W, self.attention_hidden_channels)  # (BS, num_rois*H*W, attention_hidden_channels)

        K = K.permute(0, 2, 3, 1)  # (BS*num_rois, _H, _W, attention_hidden_channels)
        K = K.reshape(BS, num_rois, _H, _W, self.attention_hidden_channels)
        K = K.reshape(BS, num_rois*_H*_W, self.attention_hidden_channels)  # (BS, num_rois*_H*_W, attention_hidden_channels)

        V = V.permute(0, 2, 3, 1)  # (BS*num_rois, _H, _W, attention_hidden_channels)
        V = V.reshape(BS, num_rois, _H, _W, self.attention_hidden_channels)
        V = V.reshape(BS, num_rois*_H*_W, self.attention_hidden_channels)  # (BS, num_rois*_H*_W, attention_hidden_channels)

        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        #print_shape(Q,K,V)
        
        WEIGHTS = torch.bmm(Q, K.permute(0, 2, 1))  # (BS, num_rois*H*W, num_rois*_H*_W)
        WEIGHTS = torch.softmax(WEIGHTS, dim=2)

        Y = torch.bmm(WEIGHTS, V)  # (BS, num_rois*H*W, attention_hidden_channels)
        Y = Y.reshape(BS, num_rois, H, W, self.attention_hidden_channels)
        Y = Y.reshape(BS*num_rois, H, W, self.attention_hidden_channels)
        Y = Y.permute(0, 3, 1, 2)
        Y = Y.contiguous()
        y = self.y_conv(Y)
        y = y.contiguous()

        if self.with_golbal:
            glb_y = self.attention(feats, x + y)
            return super(RoIAttentionConvFCBBoxHead, self).forward(glb_y)
        
        x_enhanced = x + y
        return super(RoIAttentionConvFCBBoxHead, self).forward(x_enhanced)


@HEADS.register_module()
class RoIAttentionShared2FCBBoxHead(RoIAttentionConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(RoIAttentionShared2FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class UnsharedRoIAttentionBBoxHead(RoIAttentionConvFCBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(UnsharedRoIAttentionBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=0,
            num_cls_convs=0,
            num_cls_fcs=2,
            num_reg_convs=2,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
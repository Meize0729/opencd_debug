# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import Tensor

from mmseg.utils import ConfigType, SampleList
from mmseg.models.utils import resize

from opencd.registry import MODELS
from mmdet.models.layers import DetrTransformerDecoder, SinePositionalEncoding
from opencd.evaluation.metrics import compute_metrics_tools as CMT
from PIL import Image
import numpy as np
from mmdet.models.losses import FocalLoss, CrossEntropyLoss

@MODELS.register_module()
class Foundation_FCN(BaseModule):
   
    def __init__(self,
                 channels=256,
                 out_channels=256,
                 patch_size=16,
                 img_size=512,
                 loss_type = 'FocalLoss',
                 num_classes=2,
                 positional_encoding: ConfigType = dict(
                     num_feats=128, normalize=True),
                 transformer_decoder: ConfigType = ...,
                 upsample_decoder: ConfigType = ...,
                 loss_weight=[1,1,2],
                 loss_layers=[0,1,2],
                 class_weight_bx=2.35,
                 class_weight_cd=15.77,
                 dropout_ratio=0.1,
                 act_cfg=dict(type='ReLU'),
                 train_cfg='cp building',
                 init_cfg=None):
        super().__init__(init_cfg)
        self.channels = channels
        self.patch_size = patch_size
        self.dropout_ratio = dropout_ratio
        self.act_cfg = act_cfg
        self.loss_weight = loss_weight
        self.align_corners=None

        self.train_cfg = train_cfg

        self.upsample_decoder = UpsampleDowncDecoder(
            **upsample_decoder)

        self.img_size = img_size



        self.loss_layers = loss_layers

        if loss_type == 'FocalLoss':
            self.bx = self.cd = self.all = FocalLoss(use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0)
        else:
            self.bx = CrossEntropyLoss(use_sigmoid=True, reduction='mean', class_weight=class_weight_bx)
            self.cd = CrossEntropyLoss(use_sigmoid=True, reduction='mean', class_weight=class_weight_cd)
            self.all_bx = CrossEntropyLoss(use_sigmoid=True, reduction='mean', class_weight=2.9090520662176407)
            self.all_cd = CrossEntropyLoss(use_sigmoid=True, reduction='mean', class_weight=34.17995859039499)


    def init_weights(self) -> None:
        """Initialize the weights."""
        super().init_weights()
        # 初始化上采样模块
        self.upsample_decoder.init_weights()

    def forward(self, x, batch_data_samples):
        # 获取data 信息，bs，img_shape等，用于transformer mask以及生成position embeddings

        x = x[-1]
        encoder_features_1, encoder_features_2 = torch.split(x, x.shape[0]//2, dim=0) 
        encoder_features = encoder_features_2 - encoder_features_1
        
        logits = self.upsample_decoder(encoder_features)

        return logits
    
    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs, batch_data_samples)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses

    def predict(self, inputs: Tuple[Tensor], data_samples: List[dict],
                test_cfg: ConfigType) -> Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        seg_logits = self.forward(inputs, data_samples)

        return self.predict_by_feat(seg_logits, data_samples)

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList,
                     to_metrics=True) -> dict:
        data_type = batch_data_samples[0].type
        loss_dict = dict()
        if data_type == 'only_cd_label':
            gt_semantic_segs_cd = [data_sample.gt_sem_seg.data for data_sample in batch_data_samples]
            gt_cd = torch.stack(gt_semantic_segs_cd, dim=0)

            loss_dict['loss_build_a'] = self.loss_weight[0] * seg_logits.new_zeros([1])
            loss_dict['loss_build_b'] = self.loss_weight[1] * seg_logits.new_zeros([1])
            loss_dict['loss_cd']      = self.loss_weight[2] * self.cd(seg_logits.reshape(-1,1), gt_cd.float().reshape(-1,1))

            cd_pred = seg_logits.sigmoid()
            cd_pred = (cd_pred > 0.5).float()
            gt_cd = (gt_cd > 0).float()

            loss_dict['P'], loss_dict['R'], loss_dict['F1'], loss_dict['IoU'] = torch.tensor(CMT(cd_pred, gt_cd))

        return loss_dict

    def predict_by_feat(self, seg_logits: Tensor,
                        data_samples: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """

        # seg_logits = resize(
        #     input=seg_logits,
        #     size=data_samples[0].metainfo['img_shape'],
        #     mode='nearest',
        #     align_corners=self.align_corners
        #     )
        return seg_logits

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
from .upsample_decoder import UpsampleDecoder
from opencd.evaluation.metrics import compute_metrics_tools as CMT
from PIL import Image
import numpy as np
from mmdet.models.losses import FocalLoss, CrossEntropyLoss
from .transformer import *

@MODELS.register_module()
class Foundation_Decoder_V3(BaseModule):
   
    def __init__(self,
                 channels=256,
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
                 act_cfg=dict(type='ReLU'),
                 train_cfg='cp building',
                 init_cfg=None):
        super().__init__(init_cfg)
        self.channels = channels
        self.patch_size = patch_size
        self.act_cfg = act_cfg
        self.loss_weight = loss_weight
        self.align_corners=None
        self.img_size = img_size

        self.train_cfg = train_cfg

        self.upsample_decoder_building = UpsampleDecoder(
            **upsample_decoder)
        self.upsample_decoder_cd = UpsampleDecoder(
            **upsample_decoder)
        self.transformer_decoder = TwoWayTransformer(
                depth=2,
                embedding_dim=channels,
                mlp_dim=2048,
                num_heads=8,
            ),
        self.transformer_decoder = self.transformer_decoder[0]

        self.decoder_pe = SinePositionalEncoding(**positional_encoding)
        self.building_a_embed = nn.Embedding(1, self.channels)
        self.building_b_embed = nn.Embedding(1, self.channels)
        self.cd_embed = nn.Embedding(1, self.channels)
        self.decoder_picture_embed = nn.Parameter(
            torch.zeros(2, 1, self.channels)
        )

        self.loss_layers = loss_layers
        self.loss_type = loss_type
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
        self.upsample_decoder_cd.init_weights()
        self.upsample_decoder_building.init_weights()
        # 初始化transformer模块
        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, batch_data_samples):
        # 获取data 信息，bs，img_shape等，用于transformer mask以及生成position embeddings
        
        x = x[-1]

        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        batch_size = len(batch_img_metas)

        input_img_h = input_img_w = self.img_size
        padding_mask = x[-1].new_ones((batch_size, input_img_h, input_img_w),
                                      dtype=torch.float32)
        for i in range(batch_size):
            # img_h, img_w = batch_img_metas[i]['img_shape']
            img_h, img_w = 512, 512
            padding_mask[i, :img_h, :img_w] = 0
        padding_mask = F.interpolate(
            padding_mask.unsqueeze(1), size=x.shape[-2:],
            mode='nearest').to(torch.bool).squeeze(1)

        pos_embed = self.decoder_pe(padding_mask)
        padding_mask = torch.cat((padding_mask.flatten(1), padding_mask.flatten(1)), dim=1)
        pos_embed = pos_embed.flatten(2).permute(0, 2, 1)

        # 多图情况
        if x.shape[0] != batch_size:
            # 沿batch切开，按token拼起来，[real_bs, token(2048), embed_num]
            encoder_features_1, encoder_features_2 = torch.split(x, x.shape[0]//2, dim=0) 
            encoder_features = torch.cat((encoder_features_1.flatten(2).permute(0,2,1),
                                          encoder_features_2.flatten(2).permute(0,2,1)), dim=1)
            pos_embed = torch.cat((pos_embed, pos_embed), dim=1)
            pos_embed[:, :pos_embed.shape[1]//2, :] = pos_embed[:, :pos_embed.shape[1]//2, :] + self.decoder_picture_embed[0]
            pos_embed[:, pos_embed.shape[1]//2:, :] = pos_embed[:, pos_embed.shape[1]//2:, :] + self.decoder_picture_embed[1]
        else:
            encoder_features = x.flatten(2).permute(0,2,1)
        # 拼global和query embed，按token
        global_embed_building_a, global_embed_building_b, global_embed_cd = \
            self.building_a_embed.weight, self.building_b_embed.weight, self.cd_embed.weight
        query_embed = torch.cat((global_embed_building_a, global_embed_building_b, global_embed_cd), dim=0)

        query_embed = query_embed.unsqueeze(0).repeat(batch_size, 1, 1)
        target = torch.zeros_like(query_embed)

        query_embed, image_embeddings = self.transformer_decoder(
            image_embedding=encoder_features,
            image_pe=pos_embed,
            point_embedding=query_embed
        )
        a_embed, b_embed, cd_embed = torch.split(query_embed, [1,1,1], dim=1)
        img_a, img_b = torch.split(image_embeddings, (self.img_size//self.patch_size)**2, dim=1)
        img_a = img_a.permute(0, 2, 1).reshape(batch_size, self.channels, 
                                        self.img_size//self.patch_size, self.img_size//self.patch_size)
        img_b = img_b.permute(0, 2, 1).reshape(batch_size, self.channels, 
                                        self.img_size//self.patch_size, self.img_size//self.patch_size) 
        img_cd = img_b - img_a
        mask_a = self.upsample_decoder_building(img_a)
        mask_b = self.upsample_decoder_building(img_b)
        mask_cd = self.upsample_decoder_cd(img_cd)

        mask_a_logits = torch.einsum('bqc,bchw->bqhw',a_embed, mask_a)
        mask_b_logits = torch.einsum('bqc,bchw->bqhw',b_embed, mask_b)
        mask_cd_logits = torch.einsum('bqc,bchw->bqhw',cd_embed, mask_cd)         

        logits = torch.cat((mask_a_logits, mask_b_logits, mask_cd_logits), dim=1)

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
                     batch_data_samples: SampleList) -> dict:
        data_type = batch_data_samples[0].type
        a_logits, b_logits, cd_logits = torch.split(seg_logits, [1, 1, 1], dim=1)
        loss_dict = dict()
        if data_type == 'only_cd_label':
            gt_semantic_segs_cd = [data_sample.gt_sem_seg.data for data_sample in batch_data_samples]
            gt_cd = torch.stack(gt_semantic_segs_cd, dim=0)

            cd_pred = cd_logits.sigmoid()
            cd_pred = (cd_pred > 0.5).float()
            gt_cd = (gt_cd > 0).float()
            loss_dict['P'], loss_dict['R'], loss_dict['F1'], loss_dict['IoU'] = torch.tensor(CMT(cd_pred, gt_cd))

            if self.loss_type == 'FocalLoss':
                loss_dict['loss_build_a'] = self.loss_weight[0] * seg_logits.new_zeros([1])
                loss_dict['loss_build_b'] = self.loss_weight[1] * seg_logits.new_zeros([1])
                cd_logits = torch.cat((1 - cd_logits.reshape(-1,1), cd_logits.reshape(-1,1)), dim=1)
                gt_cd = F.one_hot(gt_cd.long().reshape(-1,1).squeeze(), num_classes=2)
                loss_dict['loss_cd']      = self.loss_weight[2] * self.cd(cd_logits, gt_cd)
            else:
                loss_dict['loss_build_a'] = self.loss_weight[0] * seg_logits.new_zeros([1])
                loss_dict['loss_build_b'] = self.loss_weight[1] * seg_logits.new_zeros([1])
                loss_dict['loss_cd']      = self.loss_weight[2] * self.cd(cd_logits.reshape(-1,1), gt_cd.float().reshape(-1,1))  


        elif data_type == 'only_building_label':
            gt_semantic_segs_a = [data_sample.gt_sem_seg_from.data for data_sample in batch_data_samples]
            gt_a = torch.stack(gt_semantic_segs_a, dim=0)
            gt_b = gt_a

            a_pred = a_logits.sigmoid()
            a_pred = (a_pred > 0.5).float()
            gt_a = (gt_a > 0).float()

            loss_dict['P'], loss_dict['R'], loss_dict['F1'], loss_dict['IoU'] = torch.tensor(CMT(a_pred, gt_a))
            if self.train_cfg == 'cp building':
                gt_cd = seg_logits.new_zeros(cd_logits.shape)
                if self.loss_type == 'FocalLoss':
                    a_logits = torch.cat((1 - a_logits.reshape(-1,1), a_logits.reshape(-1,1)), dim=1)
                    b_logits = torch.cat((1 - b_logits.reshape(-1,1), b_logits.reshape(-1,1)), dim=1)
                    cd_logits = torch.cat((1 - cd_logits.reshape(-1,1), cd_logits.reshape(-1,1)), dim=1)
                    gt_a = F.one_hot(gt_a.long().reshape(-1,1).squeeze(), num_classes=2)
                    gt_b = F.one_hot(gt_b.long().reshape(-1,1).squeeze(), num_classes=2)
                    gt_cd = F.one_hot(gt_cd.long().reshape(-1,1).squeeze(), num_classes=2)
                    loss_dict['loss_build_a'] = self.loss_weight[0] * self.bx(a_logits, gt_a)
                    loss_dict['loss_build_b'] = self.loss_weight[1] * self.bx(b_logits, gt_b)
                    loss_dict['loss_cd'] = self.loss_weight[2] * self.cd(cd_logits, gt_cd) * 0.1
                else:
                    loss_dict['loss_build_a'] = self.loss_weight[0] * self.bx(a_logits.reshape(-1,1), gt_a.float().reshape(-1,1))
                    loss_dict['loss_build_b'] = self.loss_weight[1] * self.bx(b_logits.reshape(-1,1), gt_b.float().reshape(-1,1))
                    loss_dict['loss_cd'] = self.loss_weight[2] * self.cd(cd_logits.reshape(-1,1), gt_cd.float().reshape(-1,1)) * 0.1
                
                # loss_dict['0:1'] = torch.tensor( (gt_a == 0).sum().item() / max((gt_a == 1).sum().item(), 1e-9) )
        else: 
            gt_semantic_segs_cd = [data_sample.gt_sem_seg.data for data_sample in batch_data_samples]
            gt_semantic_segs_a = [data_sample.gt_sem_seg_from.data for data_sample in batch_data_samples]
            gt_semantic_segs_b = [data_sample.gt_sem_seg_to.data for data_sample in batch_data_samples]
            gt_a = torch.stack(gt_semantic_segs_a, dim=0)
            gt_b = torch.stack(gt_semantic_segs_b, dim=0)
            gt_cd = torch.stack(gt_semantic_segs_cd, dim=0)
            
            _pred = seg_logits.sigmoid()
            _pred = (_pred > 0.5).float()
            _gt = torch.cat((gt_a.unsqueeze(0), gt_b.unsqueeze(0), gt_cd.unsqueeze(0)), dim=0)
            _gt = (_gt > 0).float()

            loss_dict['P'], loss_dict['R'], loss_dict['F1'], loss_dict['IoU'] = torch.tensor(CMT(_pred, _gt))

            loss_dict['0:1'] = torch.tensor( (gt_cd == 0).sum().item() / max((gt_cd == 1).sum().item(), 1e-9) )

            if self.loss_type == 'FocalLoss':
                a_logits = torch.cat((1 - a_logits.reshape(-1,1), a_logits.reshape(-1,1)), dim=1)
                b_logits = torch.cat((1 - b_logits.reshape(-1,1), b_logits.reshape(-1,1)), dim=1)
                cd_logits = torch.cat((1 - cd_logits.reshape(-1,1), cd_logits.reshape(-1,1)), dim=1)
                gt_a = F.one_hot(gt_a.long().reshape(-1,1).squeeze(), 2)
                gt_b = F.one_hot(gt_b.long().reshape(-1,1).squeeze(), 2)
                gt_cd = F.one_hot(gt_cd.long().reshape(-1,1).squeeze(), 2)
                loss_dict['loss_build_a'] = self.loss_weight[0] * self.bx(a_logits, gt_a)
                loss_dict['loss_build_b'] = self.loss_weight[1] * self.bx(b_logits, gt_b)
                loss_dict['loss_cd'] = self.loss_weight[2] * self.cd(cd_logits, gt_cd)
            else:
                loss_dict['loss_build_a'] = self.loss_weight[0] * self.bx(a_logits.reshape(-1,1), gt_a.float().reshape(-1,1))
                loss_dict['loss_build_b'] = self.loss_weight[1] * self.bx(b_logits.reshape(-1,1), gt_b.float().reshape(-1,1))
                loss_dict['loss_cd'] = self.loss_weight[2] * self.cd(cd_logits.reshape(-1,1), gt_cd.float().reshape(-1,1))

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

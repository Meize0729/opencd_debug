# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import Tensor
import einops

from mmseg.utils import ConfigType, SampleList
from mmseg.models.utils import resize

from opencd.registry import MODELS
from mmengine.model import BaseModule, ModuleList, caffe2_xavier_init
from .upsample_decoder import UpsampleDecoder
from opencd.evaluation.metrics import compute_metrics_tools as CMT
from PIL import Image
import numpy as np
from mmdet.models.losses import FocalLoss, CrossEntropyLoss
from .transformer import *
from opencd.models.losses import EdgeLoss

@MODELS.register_module()
class Foundation_Decoder_SAM_BX_V3(BaseModule):
   
    def __init__(self,
                 transformer_dim: int,
                 activation=nn.GELU,
                 embed_head_depth: int = 3,
                 embed_head_hidden_dim: int = 256,

                 channels=256,
                 patch_size=16,
                 img_size=512,
                 decoder_layer=3,
                 loss_type = 'Normal',
                 num_classes=2,
                 loss_weight=[1,1,2],
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

        self.transformer_dim = transformer_dim
        self.transformer_decoder = TwoWayTransformer(
                depth=decoder_layer,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
                attention_downsample_rate=2
            )
        self.pe_emb = PositionEmbeddingRandom(transformer_dim // 2)

        self.output_upscaling1 = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 2, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 2),
            activation(),
        )
        
        self.output_upscaling2 = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim // 2, transformer_dim // 2, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 2),
            activation(),
        )
        
        self.output_upscaling3 = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim // 2, transformer_dim // 2, kernel_size=2, stride=2),
            activation(),
        )

        self.pred_head = MLP(
            transformer_dim, embed_head_hidden_dim, transformer_dim // 2, embed_head_depth
        )


        self.building_a_embed = nn.Embedding(1, self.channels)
        self.building_b_embed = nn.Embedding(1, self.channels)
        self.cd_embed = nn.Embedding(1, self.channels)
        self.decoder_picture_embed = nn.Parameter(
            torch.zeros(2, 1, self.channels)
        )

        self.loss_type = loss_type
        if loss_type == 'EdgeLoss':
            self.bx = self.cd = self.all = EdgeLoss()
        else:
            self.bx = CrossEntropyLoss(use_sigmoid=True, reduction='mean', ignore_index=255)
            self.cd = CrossEntropyLoss(use_sigmoid=True, reduction='mean', ignore_index=255)
            self.all_bx = CrossEntropyLoss(use_sigmoid=True, reduction='mean', ignore_index=255)
            self.all_cd = CrossEntropyLoss(use_sigmoid=True, reduction='mean', ignore_index=255)
            

    def init_weights(self) -> None:
        """Initialize the weights."""
        super().init_weights()
        # 初始化上采样模块
        caffe2_xavier_init(self.pred_head, bias=0)
        caffe2_xavier_init(self.output_upscaling1, bias=0)
        caffe2_xavier_init(self.output_upscaling2, bias=0)
        caffe2_xavier_init(self.output_upscaling3, bias=0)
        # 初始化transformer模块
        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, batch_data_samples):
        # 获取data 信息，bs，img_shape等，用于transformer mask以及生成position embeddings
        x = x[-1]

        # 沿batch切开，按token拼起来，[real_bs, token(2048), embed_num]
        encoder_features_1, encoder_features_2 = torch.split(x, x.shape[0]//2, dim=0) 
        b, c, h, w = encoder_features_1.shape
        encoder_features = torch.cat((encoder_features_1.flatten(2).permute(0,2,1),
                                        encoder_features_2.flatten(2).permute(0,2,1)), dim=1)
        

        pos_embed = self.pe_emb((h, w))
        pos_embed = einops.rearrange(pos_embed, 'c h w -> (h w) c')
        pos_embed = einops.repeat(pos_embed, 'n c -> b n c', b=b)
        pos_embed = torch.cat((pos_embed, pos_embed), dim=1)
        pos_embed[:, :pos_embed.shape[1]//2, :] = pos_embed[:, :pos_embed.shape[1]//2, :] + self.decoder_picture_embed[0]
        pos_embed[:, pos_embed.shape[1]//2:, :] = pos_embed[:, pos_embed.shape[1]//2:, :] + self.decoder_picture_embed[1]


        # 拼global和query embed，按token
        global_embed_a = self.building_a_embed.weight
        global_embed_b = self.building_b_embed.weight
        global_embed_cd = global_embed_a.new_zeros((1, self.channels))
        
        query_embed = torch.cat((global_embed_a, global_embed_b, global_embed_cd), dim=0)
        query_embed = query_embed.unsqueeze(0).repeat(b, 1, 1)


        query_embed, image_embeddings = self.transformer_decoder(
            image_embedding=encoder_features,
            image_pe=pos_embed,
            point_embedding=query_embed
        )

        a_embed, b_embed, _ = torch.split(query_embed, [1,1,1], dim=1)
        
        a_embed = self.pred_head(a_embed)
        b_embed = self.pred_head(b_embed)
        
        img_a, img_b = torch.split(image_embeddings, (self.img_size//self.patch_size)**2, dim=1)
        img_a = img_a.permute(0, 2, 1).reshape(b, self.transformer_dim, 
                                        self.img_size//self.patch_size, self.img_size//self.patch_size)
        img_b = img_b.permute(0, 2, 1).reshape(b, self.transformer_dim, 
                                        self.img_size//self.patch_size, self.img_size//self.patch_size)

        mask_a1 = self.output_upscaling1(img_a)
        mask_b1 = self.output_upscaling1(img_b)

        b, c, h, w = mask_a1.shape
        mask_a_logits = (a_embed @ mask_a1.view(b, c, h * w)).view(b, -1, h, w)
        mask_b_logits = (b_embed @ mask_b1.view(b, c, h * w)).view(b, -1, h, w)

        logits1 = torch.cat((mask_a_logits, mask_b_logits, mask_b_logits), dim=1)

        logits1 = resize(
            input=logits1,
            size=batch_data_samples[0].metainfo['img_shape'],
            mode='bilinear',
            align_corners=self.align_corners
            )

        mask_a2 = self.output_upscaling2(mask_a1)
        mask_b2 = self.output_upscaling2(mask_b1)

        b, c, h, w = mask_a2.shape
        mask_a_logits = (a_embed @ mask_a2.view(b, c, h * w)).view(b, -1, h, w)
        mask_b_logits = (b_embed @ mask_b2.view(b, c, h * w)).view(b, -1, h, w)

        logits2 = torch.cat((mask_a_logits, mask_b_logits, mask_b_logits), dim=1)

        logits2 = resize(
            input=logits2,
            size=batch_data_samples[0].metainfo['img_shape'],
            mode='bilinear',
            align_corners=self.align_corners
            )        
        
        mask_a3 = self.output_upscaling3(mask_a2)
        mask_b3 = self.output_upscaling3(mask_b2)

        b, c, h, w = mask_a3.shape
        mask_a_logits = (a_embed @ mask_a3.view(b, c, h * w)).view(b, -1, h, w)
        mask_b_logits = (b_embed @ mask_b3.view(b, c, h * w)).view(b, -1, h, w)

        logits3 = torch.cat((mask_a_logits, mask_b_logits, mask_b_logits), dim=1)

        logits3 = resize(
            input=logits2,
            size=batch_data_samples[0].metainfo['img_shape'],
            mode='bilinear',
            align_corners=self.align_corners
            )      

        return (logits1, logits2, logits3)
    
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
        a_logits_1, b_logits_1, cd_logits_1 = torch.split(seg_logits[0], [1, 1, 1], dim=1)
        a_logits_2, b_logits_2, cd_logits_2 = torch.split(seg_logits[1], [1, 1, 1], dim=1)
        a_logits_3, b_logits_3, cd_logits_3 = torch.split(seg_logits[2], [1, 1, 1], dim=1)
        loss_dict = dict()

        gt_semantic_segs_a = [data_sample.gt_sem_seg_from.data for data_sample in batch_data_samples]
        gt_a = torch.stack(gt_semantic_segs_a, dim=0)

        a_pred = a_logits_3.sigmoid()
        a_pred = (a_pred > 0.5).float()
        gt_a = (gt_a > 0).float()

        loss_dict['P'], loss_dict['R'], loss_dict['F1'], loss_dict['IoU'] = torch.tensor(CMT(a_pred, gt_a))

        gt_cd = seg_logits[2].new_zeros(cd_logits_3.shape)
        gt_b = gt_a
        
        loss_dict['loss_build_a_1'] = self.loss_weight[0] * self.bx(a_logits_1, gt_a.float())
        loss_dict['loss_build_b_1'] = self.loss_weight[1] * self.bx(b_logits_1, gt_b.float())
        loss_dict['loss_build_a_2'] = self.loss_weight[0] * self.bx(a_logits_2, gt_a.float())
        loss_dict['loss_build_b_2'] = self.loss_weight[1] * self.bx(b_logits_2, gt_b.float())
        loss_dict['loss_build_a'] = self.loss_weight[0] * self.bx(a_logits_3, gt_a.float())
        loss_dict['loss_build_b'] = self.loss_weight[1] * self.bx(b_logits_3, gt_b.float())
        loss_dict['loss_cd'] = self.loss_weight[2] * seg_logits[2].new_zeros([1])

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
        return seg_logits[-1]

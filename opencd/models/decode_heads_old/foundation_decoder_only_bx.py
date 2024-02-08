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
from .upsample_fpn_head import UpsampleFPNHead
from opencd.evaluation.metrics import compute_metrics_tools as CMT
from PIL import Image
import numpy as np
from mmdet.models.losses import FocalLoss, CrossEntropyLoss

from opencd.models.losses import EdgeLoss
from .transformer import *

@MODELS.register_module()
class Foundation_Decoder_only_bx(BaseModule):
   
    def __init__(self,
                 transformer_dim: int,
                 activation=nn.GELU,
                 embed_head_depth: int = 3,
                 embed_head_hidden_dim: int = 256,
                 decoder_layer: int = 3,

                 channels=256,
                 patch_size=16,
                 img_size=512,
                 loss_type = 'FocalLoss',
                 num_classes=2,
                 loss_weight=[1,1,1],
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
            )
        self.pe_emb = PositionEmbeddingRandom(transformer_dim // 2)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim),
            activation(),
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 2, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 2),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 2, transformer_dim // 4, kernel_size=2, stride=2),
            activation(),
        )
        self.pred_head = MLP(
            transformer_dim, embed_head_hidden_dim, transformer_dim // 4, embed_head_depth
        )
        self.building_a_embed = nn.Embedding(1, self.channels)

        self.loss_type = loss_type
        if loss_type == 'EdgeLoss':
            self.bx = self.cd = self.all = EdgeLoss(edge_factor=10)
        else:
            self.bx = CrossEntropyLoss(use_sigmoid=True, reduction='mean', ignore_index=255)
            self.cd = CrossEntropyLoss(use_sigmoid=True, reduction='mean', ignore_index=255)
            self.all_bx = CrossEntropyLoss(use_sigmoid=True, reduction='mean', ignore_index=255)
            self.all_cd = CrossEntropyLoss(use_sigmoid=True, reduction='mean', ignore_index=255)

        # self.bx = EdgeLoss()

    def init_weights(self) -> None:
        """Initialize the weights."""
        super().init_weights()
        # 初始化上采样模块
        caffe2_xavier_init(self.pred_head, bias=0)
        caffe2_xavier_init(self.output_upscaling, bias=0)
        # 初始化transformer模块
        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, batch_data_samples):
        # import pdb; pdb.set_trace()
        # 获取data 信息，bs，img_shape等，用于transformer mask以及生成position embeddings
        x = x[-1]

        encoder_features = x
        b, c, h, w = encoder_features.shape
        pos_embed = self.pe_emb((h, w))
        pos_embed = einops.rearrange(pos_embed, 'c h w -> (h w) c')
        pos_embed = einops.repeat(pos_embed, 'n c -> b n c', b=b)
        encoder_features = encoder_features.flatten(2).permute(0,2,1)
        # 拼global和query embed，按token
        global_embed_a = self.building_a_embed.weight
        
        query_embed = global_embed_a
        query_embed = query_embed.unsqueeze(0).repeat(b, 1, 1)

        query_embed, image_embeddings = self.transformer_decoder(
            image_embedding=encoder_features,
            image_pe=pos_embed,
            point_embedding=query_embed
        )

        a_embed = query_embed
        a_embed = self.pred_head(a_embed)
        
        img_a = image_embeddings
        img_a = img_a.permute(0, 2, 1).reshape(b, self.transformer_dim, 
                                        self.img_size//self.patch_size, self.img_size//self.patch_size)

        mask_a = self.output_upscaling(img_a)

        b, c, h, w = mask_a.shape
        mask_a_logits = (a_embed @ mask_a.view(b, c, h * w)).view(b, -1, h, w)

        logits = mask_a_logits

        logits = resize(
            input=logits,
            size=batch_data_samples[0].metainfo['img_shape'],
            mode='bilinear',
            align_corners=self.align_corners
            )

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
        a_logits = seg_logits
        loss_dict = dict()
        try:
            gt_semantic_segs_a = [data_sample.gt_sem_seg_from.data for data_sample in batch_data_samples]
        except:
            gt_semantic_segs_a = [data_sample.gt_sem_seg.data for data_sample in batch_data_samples]
        gt_a = torch.stack(gt_semantic_segs_a, dim=0)
        loss_dict['loss_build_a'] = self.loss_weight[0] * self.bx(a_logits, gt_a.float())
        a_pred = a_logits.sigmoid()
        a_pred = (a_pred > 0.5).float()
        gt_a = (gt_a > 0).float()

        loss_dict['P'], loss_dict['R'], loss_dict['F1'], loss_dict['IoU'] = torch.tensor(CMT(a_pred, gt_a))
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


@MODELS.register_module()
class Foundation_Decoder_only_bx_hqfpn(BaseModule):
   
    def __init__(self,
                 transformer_dim: int,
                 activation=nn.GELU,
                 embed_head_depth: int = 3,
                 embed_head_hidden_dim: int = 256,
                 decoder_layer: int = 3,

                 channels=256,
                 patch_size=16,
                 img_size=512,
                 loss_type = 'FocalLoss',
                 num_classes=2,
                 loss_weight=[1,1,1],
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
            )

        self.compress_vit = nn.Sequential(
            nn.ConvTranspose2d(768, transformer_dim, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim),
            activation(),
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 2, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 2),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 2, transformer_dim // 4, kernel_size=2, stride=2),
            activation(),
        )

        self.embeddings_encoder = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim),
            activation(),
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 2, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 2),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 2, transformer_dim // 4, kernel_size=2, stride=2),
            activation(),
        )


        self.pe_emb = PositionEmbeddingRandom(transformer_dim // 2)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim),
            activation(),
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 2, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 2),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 2, transformer_dim // 4, kernel_size=2, stride=2),
            activation(),
        )

        self.pred_head = MLP(
            transformer_dim, embed_head_hidden_dim, transformer_dim // 4, embed_head_depth
        )
        self.building_a_embed = nn.Embedding(1, self.channels)

        self.loss_type = loss_type
        if loss_type == 'EdgeLoss':
            self.bx = self.cd = self.all = EdgeLoss(edge_factor=10)
        else:
            self.bx = CrossEntropyLoss(use_sigmoid=True, reduction='mean', ignore_index=255)
            self.cd = CrossEntropyLoss(use_sigmoid=True, reduction='mean', ignore_index=255)
            self.all_bx = CrossEntropyLoss(use_sigmoid=True, reduction='mean', ignore_index=255)
            self.all_cd = CrossEntropyLoss(use_sigmoid=True, reduction='mean', ignore_index=255)

        # self.bx = EdgeLoss()

    def init_weights(self) -> None:
        """Initialize the weights."""
        super().init_weights()
        # 初始化上采样模块
        caffe2_xavier_init(self.pred_head, bias=0)
        caffe2_xavier_init(self.output_upscaling, bias=0)
        caffe2_xavier_init(self.compress_vit, bias=0)
        caffe2_xavier_init(self.embeddings_encoder, bias=0)
        # 初始化transformer模块
        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, batch_data_samples):
        # import pdb; pdb.set_trace()
        # 获取data 信息，bs，img_shape等，用于transformer mask以及生成position embeddings
        hq_features = self.embeddings_encoder(x[-1]) + self.compress_vit(x[0])

        encoder_features = x[-1]
        b, c, h, w = encoder_features.shape
        pos_embed = self.pe_emb((h, w))
        pos_embed = einops.rearrange(pos_embed, 'c h w -> (h w) c')
        pos_embed = einops.repeat(pos_embed, 'n c -> b n c', b=b)
        encoder_features = encoder_features.flatten(2).permute(0,2,1)
        # 拼global和query embed，按token
        global_embed_a = self.building_a_embed.weight
        
        query_embed = global_embed_a
        query_embed = query_embed.unsqueeze(0).repeat(b, 1, 1)

        query_embed, image_embeddings = self.transformer_decoder(
            image_embedding=encoder_features,
            image_pe=pos_embed,
            point_embedding=query_embed
        )

        a_embed = query_embed
        a_embed = self.pred_head(a_embed)
        
        img_a = image_embeddings
        img_a = img_a.permute(0, 2, 1).reshape(b, self.transformer_dim, 
                                        self.img_size//self.patch_size, self.img_size//self.patch_size)

        mask_a = self.output_upscaling(img_a) + hq_features

        b, c, h, w = mask_a.shape
        mask_a_logits = (a_embed @ mask_a.view(b, c, h * w)).view(b, -1, h, w)

        logits = mask_a_logits

        logits = resize(
            input=logits,
            size=batch_data_samples[0].metainfo['img_shape'],
            mode='bilinear',
            align_corners=self.align_corners
            )

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
        a_logits = seg_logits
        loss_dict = dict()
        try:
            gt_semantic_segs_a = [data_sample.gt_sem_seg_from.data for data_sample in batch_data_samples]
        except:
            gt_semantic_segs_a = [data_sample.gt_sem_seg.data for data_sample in batch_data_samples]
        gt_a = torch.stack(gt_semantic_segs_a, dim=0)
        loss_dict['loss_build_a'] = self.loss_weight[0] * self.bx(a_logits, gt_a.float())
        a_pred = a_logits.sigmoid()
        a_pred = (a_pred > 0.5).float()
        gt_a = (gt_a > 0).float()

        loss_dict['P'], loss_dict['R'], loss_dict['F1'], loss_dict['IoU'] = torch.tensor(CMT(a_pred, gt_a))
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


@MODELS.register_module()
class Foundation_Decoder_only_bx_decoderfpn(BaseModule):
   
    def __init__(self,
                 transformer_dim: int,
                 activation=nn.GELU,
                 embed_head_depth: int = 3,
                 embed_head_hidden_dim: int = 256,
                 decoder_layer: int = 3,

                 channels=256,
                 patch_size=16,
                 img_size=512,
                 loss_type = 'FocalLoss',
                 num_classes=2,
                 loss_weight=[1,1,1],
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
        self.transformer_decoder = TwoWayTransformer_fpn(
                depth=4,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            )
        self.pe_emb = PositionEmbeddingRandom(transformer_dim // 2)

        self.output_upscaling = UpsampleFPNHead()

        self.pred_head = MLP(
            transformer_dim, embed_head_hidden_dim, transformer_dim, embed_head_depth
        )
        self.building_a_embed = nn.Embedding(1, self.channels)

        self.loss_type = loss_type
        if loss_type == 'EdgeLoss':
            self.bx = self.cd = self.all = EdgeLoss(edge_factor=10)
        else:
            self.bx = CrossEntropyLoss(use_sigmoid=True, reduction='mean', ignore_index=255)
            self.cd = CrossEntropyLoss(use_sigmoid=True, reduction='mean', ignore_index=255)
            self.all_bx = CrossEntropyLoss(use_sigmoid=True, reduction='mean', ignore_index=255)
            self.all_cd = CrossEntropyLoss(use_sigmoid=True, reduction='mean', ignore_index=255)

        # self.bx = EdgeLoss()

    def init_weights(self) -> None:
        """Initialize the weights."""
        super().init_weights()
        # 初始化上采样模块
        caffe2_xavier_init(self.pred_head, bias=0)
        caffe2_xavier_init(self.output_upscaling, bias=0)
        # 初始化transformer模块
        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, batch_data_samples):
        # import pdb; pdb.set_trace()
        # 获取data 信息，bs，img_shape等，用于transformer mask以及生成position embeddings
        # x = x[-1]
        encoder_features = []
        for feature in x:
            encoder_features.append(feature.flatten(2).permute(0,2,1))
        b, c, h, w = x[-1].shape
        pos_embed = self.pe_emb((h, w))
        pos_embed = einops.rearrange(pos_embed, 'c h w -> (h w) c')
        pos_embed = einops.repeat(pos_embed, 'n c -> b n c', b=b)
        # encoder_features = encoder_features.flatten(2).permute(0,2,1)
        # 拼global和query embed，按token
        global_embed_a = self.building_a_embed.weight
        
        query_embed = global_embed_a
        query_embed = query_embed.unsqueeze(0).repeat(b, 1, 1)

        query_embed, image_embeddings = self.transformer_decoder(
            image_embedding=encoder_features,
            image_pe=pos_embed,
            point_embedding=query_embed
        )

        a_embed = query_embed
        a_embed = self.pred_head(a_embed)
        
        for i in range(len(image_embeddings)):
            image_embeddings[i] = image_embeddings[i].permute(0, 2, 1).reshape(b, self.transformer_dim, 
                                        self.img_size//self.patch_size, self.img_size//self.patch_size)
        # img_a = img_a.permute(0, 2, 1).reshape(b, self.transformer_dim, 
        #                                 self.img_size//self.patch_size, self.img_size//self.patch_size)
        # import pdb; pdb.set_trace()
        mask_a = self.output_upscaling(image_embeddings)

        b, c, h, w = mask_a.shape
        mask_a_logits = (a_embed @ mask_a.view(b, c, h * w)).view(b, -1, h, w)

        logits = mask_a_logits

        logits = resize(
            input=logits,
            size=batch_data_samples[0].metainfo['img_shape'],
            mode='bilinear',
            align_corners=self.align_corners
            )

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
        a_logits = seg_logits
        loss_dict = dict()
        try:
            gt_semantic_segs_a = [data_sample.gt_sem_seg_from.data for data_sample in batch_data_samples]
        except:
            gt_semantic_segs_a = [data_sample.gt_sem_seg.data for data_sample in batch_data_samples]
        gt_a = torch.stack(gt_semantic_segs_a, dim=0)
        loss_dict['loss_build_a'] = self.loss_weight[0] * self.bx(a_logits, gt_a.float())
        a_pred = a_logits.sigmoid()
        a_pred = (a_pred > 0.5).float()
        gt_a = (gt_a > 0).float()

        loss_dict['P'], loss_dict['R'], loss_dict['F1'], loss_dict['IoU'] = torch.tensor(CMT(a_pred, gt_a))
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


@MODELS.register_module()
class Foundation_Decoder_only_bx_simplefpn(BaseModule):
   
    def __init__(self,
                 transformer_dim: int,
                 activation=nn.GELU,
                 embed_head_depth: int = 3,
                 embed_head_hidden_dim: int = 256,
                 decoder_layer: int = 3,

                 channels=256,
                 patch_size=16,
                 img_size=512,
                 loss_type = 'FocalLoss',
                 num_classes=2,
                 loss_weight=[1,1,1],
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
                embedding_dim=transformer_dim,
                mlp_dim=2048,
                num_heads=8,
            )
        self.pe_emb = PositionEmbeddingRandom(transformer_dim // 2)

        fpn_cfg = dict(
            type='SimpleFPN',
            in_dim=768,
            out_dims=[128, 256, 512, 1024],
        )
        fusion_cfg = dict(
            type='FusionHead',
            out_channels=256, 
            in_channels=[128, 256, 512, 1024],
        )
        self.output_upscaling_fpn = MODELS.build(fpn_cfg)
        self.fusion = MODELS.build(fusion_cfg)

        self.pred_head = MLP(
            transformer_dim, transformer_dim // 2, 256, embed_head_depth
        )
        self.building_a_embed = nn.Embedding(1, self.channels)

        self.loss_type = loss_type
        if loss_type == 'EdgeLoss':
            self.bx = self.cd = self.all = EdgeLoss(edge_factor=10)
        else:
            self.bx = CrossEntropyLoss(use_sigmoid=True, reduction='mean', ignore_index=255)
            self.cd = CrossEntropyLoss(use_sigmoid=True, reduction='mean', ignore_index=255)
            self.all_bx = CrossEntropyLoss(use_sigmoid=True, reduction='mean', ignore_index=255)
            self.all_cd = CrossEntropyLoss(use_sigmoid=True, reduction='mean', ignore_index=255)

        # self.bx = EdgeLoss()

    def init_weights(self) -> None:
        """Initialize the weights."""
        super().init_weights()
        # 初始化上采样模块
        caffe2_xavier_init(self.pred_head, bias=0)
        # caffe2_xavier_init(self.output_upscaling_fpn, bias=0)
        # 初始化transformer模块
        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, batch_data_samples):
        # import pdb; pdb.set_trace()
        # 获取data 信息，bs，img_shape等，用于transformer mask以及生成position embeddings
        x = x[-1]

        encoder_features = x
        b, c, h, w = encoder_features.shape
        pos_embed = self.pe_emb((h, w))
        pos_embed = einops.rearrange(pos_embed, 'c h w -> (h w) c')
        pos_embed = einops.repeat(pos_embed, 'n c -> b n c', b=b)
        encoder_features = encoder_features.flatten(2).permute(0,2,1)
        # 拼global和query embed，按token
        global_embed_a = self.building_a_embed.weight
        
        query_embed = global_embed_a
        query_embed = query_embed.unsqueeze(0).repeat(b, 1, 1)

        query_embed, image_embeddings = self.transformer_decoder(
            image_embedding=encoder_features,
            image_pe=pos_embed,
            point_embedding=query_embed
        )

        a_embed = query_embed
        a_embed = self.pred_head(a_embed)
        
        img_a = image_embeddings
        img_a = img_a.permute(0, 2, 1).reshape(b, self.transformer_dim, 
                                        self.img_size//self.patch_size, self.img_size//self.patch_size)

        fpn_a = self.output_upscaling_fpn(img_a)
        mask_a = self.fusion(fpn_a)

        b, c, h, w = mask_a.shape
        mask_a_logits = (a_embed @ mask_a.view(b, c, h * w)).view(b, -1, h, w)

        logits = mask_a_logits

        logits = resize(
            input=logits,
            size=batch_data_samples[0].metainfo['img_shape'],
            mode='bilinear',
            align_corners=self.align_corners
            )

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
        a_logits = seg_logits
        loss_dict = dict()
        try:
            gt_semantic_segs_a = [data_sample.gt_sem_seg_from.data for data_sample in batch_data_samples]
        except:
            gt_semantic_segs_a = [data_sample.gt_sem_seg.data for data_sample in batch_data_samples]
        gt_a = torch.stack(gt_semantic_segs_a, dim=0)
        loss_dict['loss_build_a'] = self.loss_weight[0] * self.bx(a_logits, gt_a.float())
        a_pred = a_logits.sigmoid()
        a_pred = (a_pred > 0.5).float()
        gt_a = (gt_a > 0).float()

        loss_dict['P'], loss_dict['R'], loss_dict['F1'], loss_dict['IoU'] = torch.tensor(CMT(a_pred, gt_a))
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


@MODELS.register_module()
class Foundation_Decoder_only_bx_simplefpn_qian(BaseModule):
   
    def __init__(self,
                 transformer_dim: int,
                 activation=nn.GELU,
                 embed_head_depth: int = 3,
                 embed_head_hidden_dim: int = 256,
                 decoder_layer: int = 3,

                 channels=256,
                 patch_size=16,
                 img_size=512,
                 loss_type = 'FocalLoss',
                 num_classes=2,
                 loss_weight=[1,1,1],
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
                embedding_dim=transformer_dim,
                mlp_dim=2048,
                num_heads=8,
            )
        self.pe_emb = PositionEmbeddingRandom(transformer_dim // 2)

        fpn_cfg = dict(
            type='SimpleFPN',
            in_dim=768,
            out_dims=[128, 256, 512, 1024],
        )
        fusion_cfg = dict(
            type='FusionHead',
            out_channels=256, 
            out_size_index=0,
            in_channels=[128, 256, 512, 1024],
        )
        self.output_upscaling_fpn = MODELS.build(fpn_cfg)
        self.fusion = MODELS.build(fusion_cfg)

        self.pred_head = MLP(
            transformer_dim, 128, 64, embed_head_depth
        )
        self.building_a_embed = nn.Embedding(1, self.channels)

        self.output_upscaling = nn.Sequential(
            nn.Conv2d(transformer_dim, transformer_dim // 2, kernel_size=1, stride=1),
            # nn.ConvTranspose2d(transformer_dim, transformer_dim // 2, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim//2)  ,
            activation(),
            nn.Conv2d(transformer_dim // 2, transformer_dim // 4, kernel_size=1, stride=1),
            # nn.ConvTranspose2d(transformer_dim//2, transformer_dim // 4, kernel_size=2, stride=2),
            activation(),
        )


        self.loss_type = loss_type
        if loss_type == 'EdgeLoss':
            self.bx = self.cd = self.all = EdgeLoss(edge_factor=10)
        else:
            self.bx = CrossEntropyLoss(use_sigmoid=True, reduction='mean', ignore_index=255)
            self.cd = CrossEntropyLoss(use_sigmoid=True, reduction='mean', ignore_index=255)
            self.all_bx = CrossEntropyLoss(use_sigmoid=True, reduction='mean', ignore_index=255)
            self.all_cd = CrossEntropyLoss(use_sigmoid=True, reduction='mean', ignore_index=255)

        # self.bx = EdgeLoss()

    def init_weights(self) -> None:
        """Initialize the weights."""
        super().init_weights()
        # 初始化上采样模块
        caffe2_xavier_init(self.pred_head, bias=0)
        # caffe2_xavier_init(self.output_upscaling_fpn, bias=0)
        # 初始化transformer模块
        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, batch_data_samples):
        # import pdb; pdb.set_trace()
        # 获取data 信息，bs，img_shape等，用于transformer mask以及生成position embeddings
        x = x[-1]
        x = self.output_upscaling_fpn(x)
        x = self.fusion(x)

        encoder_features = x
        b, c, h, w = encoder_features.shape
        pos_embed = self.pe_emb((h, w))
        pos_embed = einops.rearrange(pos_embed, 'c h w -> (h w) c')
        pos_embed = einops.repeat(pos_embed, 'n c -> b n c', b=b)
        encoder_features = encoder_features.flatten(2).permute(0,2,1)
        # 拼global和query embed，按token
        global_embed_a = self.building_a_embed.weight
        
        query_embed = global_embed_a
        query_embed = query_embed.unsqueeze(0).repeat(b, 1, 1)

        query_embed, image_embeddings = self.transformer_decoder(
            image_embedding=encoder_features,
            image_pe=pos_embed,
            point_embedding=query_embed
        )

        a_embed = query_embed
        a_embed = self.pred_head(a_embed)
        
        img_a = image_embeddings
        img_a = img_a.permute(0, 2, 1).reshape(b, self.transformer_dim, 
                                        self.img_size//self.patch_size*4, self.img_size//self.patch_size*4)

        # fpn_a = self.output_upscaling_fpn(img_a)
        # mask_a = self.fusion(fpn_a)
        mask_a = self.output_upscaling(img_a)

        b, c, h, w = mask_a.shape
        mask_a_logits = (a_embed @ mask_a.view(b, c, h * w)).view(b, -1, h, w)

        logits = mask_a_logits

        logits = resize(
            input=logits,
            size=batch_data_samples[0].metainfo['img_shape'],
            mode='bilinear',
            align_corners=self.align_corners
            )

        return logits
    
    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:

        seg_logits = self.forward(inputs, batch_data_samples)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses

    def predict(self, inputs: Tuple[Tensor], data_samples: List[dict],
                test_cfg: ConfigType) -> Tensor:

        seg_logits = self.forward(inputs, data_samples)

        return self.predict_by_feat(seg_logits, data_samples)

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        a_logits = seg_logits
        loss_dict = dict()
        try:
            gt_semantic_segs_a = [data_sample.gt_sem_seg_from.data for data_sample in batch_data_samples]
        except:
            gt_semantic_segs_a = [data_sample.gt_sem_seg.data for data_sample in batch_data_samples]
        gt_a = torch.stack(gt_semantic_segs_a, dim=0)
        loss_dict['loss_build_a'] = self.loss_weight[0] * self.bx(a_logits, gt_a.float())
        a_pred = a_logits.sigmoid()
        a_pred = (a_pred > 0.5).float()
        gt_a = (gt_a > 0).float()

        loss_dict['P'], loss_dict['R'], loss_dict['F1'], loss_dict['IoU'] = torch.tensor(CMT(a_pred, gt_a))
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


@MODELS.register_module()
class Foundation_Decoder_only_bx_simplefpn_qian_swin(BaseModule):
   
    def __init__(self,
                 in_channels=[128, 256, 512, 1024],
                 out_channels=256,
                #  out_dims=[128, 256, 512, 1024],
                 drop=0.0,
                 loss_type = 'FocalLoss',
                 num_classes=2,
                 loss_weight=[1,1,1],
                 init_cfg=None):
        super().__init__(init_cfg)
        self.loss_weight = loss_weight
        self.align_corners=None

        self.transformer_decoder = TwoWayTransformer_fpn_swin(
                depth=4,
                embedding_dim=[out_channels] * 4,
                mlp_dim=2048,
                num_heads=8,
            )
            
        fpn_cfg = dict(
            type='mmdet.FPN',
            in_channels=in_channels,
            out_channels=out_channels,
            start_level=0,
            add_extra_convs='on_output',
            num_outs=4,
        )
        fusion_cfg = dict(
            type='FusionHead',
            out_channels=out_channels, 
            out_size_index=0,
            in_channels=[out_channels] * 4,
        )

        self.simple_fpn = MODELS.build(fpn_cfg)
        self.fusion = MODELS.build(fusion_cfg)

        self.pred_head = MLP(
            out_channels, out_channels//4, out_channels//4, 3, drop
        )
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels//2, kernel_size=1, stride=1),
            LayerNorm2d(out_channels//2)  ,
            nn.GELU(),
            nn.ConvTranspose2d(out_channels//2, out_channels//4, kernel_size=1, stride=1),
            nn.GELU(),
        )

        self.pe_emb = PositionEmbeddingRandom(out_channels // 2)
        self.building_a_embed = nn.Embedding(1, out_channels)

        self.loss_type = loss_type
        if loss_type == 'EdgeLoss':
            self.bx = self.cd = self.all = EdgeLoss(edge_factor=10)
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
        caffe2_xavier_init(self.simple_fpn, bias=0)
        # 初始化transformer模块
        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, batch_data_samples):
        # import pdb; pdb.set_trace()
        # 获取data 信息，bs，img_shape等，用于transformer mask以及生成position embeddings\
        x = self.simple_fpn(x)
        # import pdb; pdb.set_trace()
        encoder_features = []
        pos_embed = []

        for i in range(len(x)):
            x_i = x[i]
            b, c, h, w = x_i.shape
            pos_embed_i = self.pe_emb((h, w)).to(x_i.device)
            pos_embed_i = einops.rearrange(pos_embed_i, 'c h w -> (h w) c')
            # pos_embed_i = F.interpolate(pos_embed_i.unsqueeze(1), scale_factor= 2**(-len(x)+i+1), mode='area').squeeze(1)
            # import pdb; pdb.set_trace()
            pos_embed_i = einops.repeat(pos_embed_i, 'n c -> b n c', b=b)
            x_i = x_i.flatten(2).permute(0,2,1)

            encoder_features.append(x_i)
            pos_embed.append(pos_embed_i)


        # 拼global和query embed，按token
        global_embed_a = self.building_a_embed.weight
        
        query_embed = global_embed_a
        query_embed = query_embed.unsqueeze(0).repeat(b, 1, 1)


        query_embed, image_embeddings = self.transformer_decoder(
            image_embedding=encoder_features[::-1],
            image_pe=pos_embed[::-1],
            point_embedding=query_embed
        )


        a_embed = query_embed

        a_embed = self.pred_head(a_embed)
        
        img_a = self.fusion(image_embeddings[::-1])

        mask_a = self.output_upscaling(img_a)

        b, c, h, w = mask_a.shape
        mask_a_logits = (a_embed @ mask_a.view(b, c, h * w)).view(b, -1, h, w)

        logits = mask_a_logits

        logits = resize(
            input=logits,
            size=batch_data_samples[0].metainfo['img_shape'],
            mode='bilinear',
            align_corners=self.align_corners
            )

        return logits
    
    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:

        seg_logits = self.forward(inputs, batch_data_samples)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses

    def predict(self, inputs: Tuple[Tensor], data_samples: List[dict],
                test_cfg: ConfigType) -> Tensor:

        seg_logits = self.forward(inputs, data_samples)

        return self.predict_by_feat(seg_logits, data_samples)

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        a_logits = seg_logits
        loss_dict = dict()
        try:
            gt_semantic_segs_a = [data_sample.gt_sem_seg_from.data for data_sample in batch_data_samples]
        except:
            gt_semantic_segs_a = [data_sample.gt_sem_seg.data for data_sample in batch_data_samples]
        gt_a = torch.stack(gt_semantic_segs_a, dim=0)
        loss_dict['loss_build_a'] = self.loss_weight[0] * self.bx(a_logits, gt_a.float())
        a_pred = a_logits.sigmoid()
        a_pred = (a_pred > 0.5).float()
        gt_a = (gt_a > 0).float()

        loss_dict['P'], loss_dict['R'], loss_dict['F1'], loss_dict['IoU'] = torch.tensor(CMT(a_pred, gt_a))
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

@MODELS.register_module()
class Foundation_Decoder_only_bx_simplefpn_qian_v2(BaseModule):
   
    def __init__(self,
                 in_channels=768,
                 out_channels=256,
                 out_dims=[128, 256, 512, 1024],
                 drop=0.0,
                 loss_type = 'FocalLoss',
                 num_classes=2,
                 loss_weight=[1,1,1],
                 init_cfg=None):
        super().__init__(init_cfg)
        self.loss_weight = loss_weight
        self.align_corners=None

        self.transformer_decoder = TwoWayTransformer_fpn(
                depth=len(out_dims),
                embedding_dim=out_dims[::-1],
                mlp_dim=2048,
                num_heads=8,
            )
            
        fpn_cfg = dict(
            type='SimpleFPN',
            in_dim=in_channels,
            out_dims=out_dims,
        )
        fusion_cfg = dict(
            type='FusionHead',
            out_channels=out_channels, 
            out_size_index=0,
            in_channels=out_dims,
        )
        self.simple_fpn = MODELS.build(fpn_cfg)
        self.fusion = MODELS.build(fusion_cfg)

        self.pred_head = MLP(
            out_dims[0]//2, out_channels//4, out_channels//4, 3, drop
        )
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels//2, kernel_size=1, stride=1),
            LayerNorm2d(out_channels//2)  ,
            nn.GELU(),
            nn.ConvTranspose2d(out_channels//2, out_channels//4, kernel_size=1, stride=1),
            nn.GELU(),
        )

        self.pe_emb = PositionEmbeddingRandom(out_dims[-1] // 2)
        self.building_a_embed = nn.Embedding(1, out_dims[-1])

        self.loss_type = loss_type
        if loss_type == 'EdgeLoss':
            self.bx = self.cd = self.all = EdgeLoss(edge_factor=10)
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
        caffe2_xavier_init(self.simple_fpn, bias=0)
        # 初始化transformer模块
        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, batch_data_samples):
        # import pdb; pdb.set_trace()
        # 获取data 信息，bs，img_shape等，用于transformer mask以及生成position embeddings
        x = x[-1]
        x = self.simple_fpn(x)
        encoder_features = []
        pos_embed = []

        for i in range(len(x)):
            x_i = x[i]
            b, c, h, w = x_i.shape
            pos_embed_i = self.pe_emb((h, w)).to(x_i.device)
            pos_embed_i = einops.rearrange(pos_embed_i, 'c h w -> (h w) c')
            pos_embed_i = F.interpolate(pos_embed_i.unsqueeze(1), scale_factor= 2**(-len(x)+i+1), mode='area').squeeze(1)
            # import pdb; pdb.set_trace()
            pos_embed_i = einops.repeat(pos_embed_i, 'n c -> b n c', b=b)
            x_i = x_i.flatten(2).permute(0,2,1)

            encoder_features.append(x_i)
            pos_embed.append(pos_embed_i)


        # 拼global和query embed，按token
        global_embed_a = self.building_a_embed.weight
        
        query_embed = global_embed_a
        query_embed = query_embed.unsqueeze(0).repeat(b, 1, 1)


        query_embed, image_embeddings = self.transformer_decoder(
            image_embedding=encoder_features[::-1],
            image_pe=pos_embed[::-1],
            point_embedding=query_embed
        )


        a_embed = query_embed

        a_embed = self.pred_head(a_embed)
        
        img_a = self.fusion(image_embeddings[::-1])

        mask_a = self.output_upscaling(img_a)

        b, c, h, w = mask_a.shape
        mask_a_logits = (a_embed @ mask_a.view(b, c, h * w)).view(b, -1, h, w)

        logits = mask_a_logits

        logits = resize(
            input=logits,
            size=batch_data_samples[0].metainfo['img_shape'],
            mode='bilinear',
            align_corners=self.align_corners
            )

        return logits
    
    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:

        seg_logits = self.forward(inputs, batch_data_samples)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses

    def predict(self, inputs: Tuple[Tensor], data_samples: List[dict],
                test_cfg: ConfigType) -> Tensor:

        seg_logits = self.forward(inputs, data_samples)

        return self.predict_by_feat(seg_logits, data_samples)

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        a_logits = seg_logits
        loss_dict = dict()
        try:
            gt_semantic_segs_a = [data_sample.gt_sem_seg_from.data for data_sample in batch_data_samples]
        except:
            gt_semantic_segs_a = [data_sample.gt_sem_seg.data for data_sample in batch_data_samples]
        gt_a = torch.stack(gt_semantic_segs_a, dim=0)
        loss_dict['loss_build_a'] = self.loss_weight[0] * self.bx(a_logits, gt_a.float())
        a_pred = a_logits.sigmoid()
        a_pred = (a_pred > 0.5).float()
        gt_a = (gt_a > 0).float()

        loss_dict['P'], loss_dict['R'], loss_dict['F1'], loss_dict['IoU'] = torch.tensor(CMT(a_pred, gt_a))
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


@MODELS.register_module()
class Foundation_Decoder_only_bx_simplefpn_qian_v3(BaseModule):
   
    def __init__(self,
                 in_channels=768,
                 out_channels=256,
                 drop=0.0,
                 loss_type = 'FocalLoss',
                 num_classes=2,
                 loss_weight=[1,1,1],
                 init_cfg=None):
        super().__init__(init_cfg)
        self.loss_weight = loss_weight
        self.align_corners=None

        self.transformer_decoder = TwoWayTransformer_fpn_swin(
                depth=4,
                embedding_dim=[out_channels] * 4,
                mlp_dim=2048,
                num_heads=8,
            )
            
        fpn_cfg = dict(
            type='SimpleFPN_det',
            backbone_channel=in_channels,
            in_channels=[in_channels//4, in_channels//2, in_channels, in_channels],
            out_channels=256,
            num_outs=4,
            norm_cfg=dict(type='LN2d', requires_grad=True))

        fusion_cfg = dict(
            type='FusionHead',
            out_channels=out_channels, 
            out_size_index=0,
            in_channels=[out_channels]*4,
        )
        self.simple_fpn = MODELS.build(fpn_cfg)
        self.fusion = MODELS.build(fusion_cfg)

        self.pred_head = MLP(
            out_channels, out_channels, out_channels//4, 3, drop
        )
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels//2, kernel_size=1, stride=1),
            LayerNorm2d(out_channels//2)  ,
            nn.GELU(),
            nn.ConvTranspose2d(out_channels//2, out_channels//4, kernel_size=1, stride=1),
            nn.GELU(),
        )

        self.pe_emb = PositionEmbeddingRandom(out_channels // 2)
        self.building_a_embed = nn.Embedding(1, out_channels)

        self.loss_type = loss_type
        if loss_type == 'EdgeLoss':
            self.bx = self.cd = self.all = EdgeLoss(edge_factor=10)
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
        caffe2_xavier_init(self.simple_fpn, bias=0)
        # 初始化transformer模块
        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, batch_data_samples):
        # import pdb; pdb.set_trace()
        # 获取data 信息，bs，img_shape等，用于transformer mask以及生成position embeddings
        x = x[-1]
        x = self.simple_fpn(x)
        encoder_features = []
        pos_embed = []

        for i in range(len(x)):
            x_i = x[i]
            b, c, h, w = x_i.shape
            pos_embed_i = self.pe_emb((h, w)).to(x_i.device)
            pos_embed_i = einops.rearrange(pos_embed_i, 'c h w -> (h w) c')
            # pos_embed_i = F.interpolate(pos_embed_i.unsqueeze(1), scale_factor= 2**(-len(x)+i+1), mode='area').squeeze(1)
            # import pdb; pdb.set_trace()
            pos_embed_i = einops.repeat(pos_embed_i, 'n c -> b n c', b=b)
            x_i = x_i.flatten(2).permute(0,2,1)

            encoder_features.append(x_i)
            pos_embed.append(pos_embed_i)


        # 拼global和query embed，按token
        global_embed_a = self.building_a_embed.weight
        
        query_embed = global_embed_a
        query_embed = query_embed.unsqueeze(0).repeat(b, 1, 1)


        query_embed, image_embeddings = self.transformer_decoder(
            image_embedding=encoder_features[::-1],
            image_pe=pos_embed[::-1],
            point_embedding=query_embed
        )


        a_embed = query_embed

        a_embed = self.pred_head(a_embed)
        
        img_a = self.fusion(image_embeddings[::-1])
        
        mask_a = self.output_upscaling(img_a)

        b, c, h, w = mask_a.shape
        mask_a_logits = (a_embed @ mask_a.view(b, c, h * w)).view(b, -1, h, w)

        logits = mask_a_logits

        logits = resize(
            input=logits,
            size=batch_data_samples[0].metainfo['img_shape'],
            mode='bilinear',
            align_corners=self.align_corners
            )

        return logits
    
    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:

        seg_logits = self.forward(inputs, batch_data_samples)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses

    def predict(self, inputs: Tuple[Tensor], data_samples: List[dict],
                test_cfg: ConfigType) -> Tensor:

        seg_logits = self.forward(inputs, data_samples)

        return self.predict_by_feat(seg_logits, data_samples)

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        a_logits = seg_logits
        loss_dict = dict()
        try:
            gt_semantic_segs_a = [data_sample.gt_sem_seg_from.data for data_sample in batch_data_samples]
        except:
            gt_semantic_segs_a = [data_sample.gt_sem_seg.data for data_sample in batch_data_samples]
        gt_a = torch.stack(gt_semantic_segs_a, dim=0)
        loss_dict['loss_build_a'] = self.loss_weight[0] * self.bx(a_logits, gt_a.float())
        a_pred = a_logits.sigmoid()
        a_pred = (a_pred > 0.5).float()
        gt_a = (gt_a > 0).float()

        loss_dict['P'], loss_dict['R'], loss_dict['F1'], loss_dict['IoU'] = torch.tensor(CMT(a_pred, gt_a))
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
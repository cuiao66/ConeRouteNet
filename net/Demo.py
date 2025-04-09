import math
import copy
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.nn as nn
from functools import partial
import numpy as np
import logging
from typing import Optional, List
from collections import OrderedDict
from timm.models.registry import register_model
from timm.models.resnet import resnet26d, resnet50d, resnet18d, resnet26, resnet50, resnet101d

from layers import HybridEmbed, GRUWaypointsPredictor, MultiPath_Generator, GRUWaypointsPredictorWithCommand, \
                    LinearWaypointsPredictor, PositionEmbeddingSine, TransformerEncoderLayer, TransformerEncoder, \
                    TransformerDecoderLayer, TransformerDecoder, YOLOBackbone, YOLOHead, \
                    GC_ViTBackBone

_logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = False



def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def build_attn_mask(mask_type):
    mask = torch.ones((151, 151), dtype=torch.bool).cuda()
    if mask_type == "seperate_all":
        mask[:50, :50] = False
        mask[50:67, 50:67] = False
        mask[67:84, 67:84] = False
        mask[84:101, 84:101] = False
        mask[101:151, 101:151] = False
    elif mask_type == "seperate_view":
        mask[:50, :50] = False
        mask[50:67, 50:67] = False
        mask[67:84, 67:84] = False
        mask[84:101, 84:101] = False
        mask[101:151, :] = False
        mask[:, 101:151] = False
    return mask


class DemoEnd2EndNet(nn.Module):
    def __init__(
        self,
        img_size=224,
        multi_view_img_size=112,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        enc_depth=6,
        dec_depth=6,
        dim_feedforward=2048,
        normalize_before=False,
        rgb_backbone_name="r26",
        num_heads=8,
        norm_layer=None,
        dropout=0.1,
        direct_concat=False,
        separate_view_attention=False,
        separate_all_attention=False,
        act_layer=None,
        weight_init="",
        freeze_num=-1,
        with_right_left_sensors=True,
        with_center_sensor=True,
        traffic_pred_head_type="det",
        waypoints_pred_head="heatmap",
        reverse_pos=True,
        use_view_embed=True,
        use_mmad_pretrain=None,
    ):
        super().__init__()
        self.traffic_pred_head_type = traffic_pred_head_type
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.reverse_pos = reverse_pos
        self.waypoints_pred_head = waypoints_pred_head
        self.with_right_left_sensors = with_right_left_sensors
        self.with_center_sensor = with_center_sensor

        self.direct_concat = direct_concat
        self.separate_view_attention = separate_view_attention
        self.separate_all_attention = separate_all_attention

        self.use_view_embed = use_view_embed

        if self.direct_concat:
            in_chans = in_chans * 4
            self.with_center_sensor = False
            self.with_right_left_sensors = False

        if self.separate_view_attention:
            self.attn_mask = build_attn_mask("seperate_view")
        elif self.separate_all_attention:
            self.attn_mask = build_attn_mask("seperate_all")
        else:
            self.attn_mask = None

        if rgb_backbone_name == "r50":
            self.rgb_backbone = resnet50d(
                pretrained=True,
                in_chans=in_chans,
                features_only=True,
                out_indices=[4],
            )
        elif rgb_backbone_name == "r26":
            self.rgb_backbone = resnet26d(
                pretrained=True,
                in_chans=in_chans,
                features_only=True,
                out_indices=[4],
            )
        elif rgb_backbone_name == "r18":
            self.rgb_backbone = resnet18d(
                pretrained=True,
                in_chans=in_chans,
                features_only=True,
                out_indices=[4],
            )
        elif rgb_backbone_name == "yolo":
            self.rgb_backbone = YOLOBackbone()
        elif rgb_backbone_name == "gc_ViT":

            self.rgb_backbone = GC_ViTBackBone()
            pass

        # rgb_embed_layer = partial(HybridEmbed, backbone=self.rgb_backbone)
        
        if use_mmad_pretrain:
            params = torch.load(use_mmad_pretrain)["state_dict"]
            updated_params = OrderedDict()
            for key in params:
                if "backbone" in key:
                    updated_params[key.replace("backbone.", "")] = params[key]
            self.rgb_backbone.load_state_dict(updated_params)

        # self.rgb_patch_embed = rgb_embed_layer(
        #     img_size=img_size,
        #     patch_size=patch_size,
        #     in_chans=in_chans,
        #     embed_dim=embed_dim,
        # )

        self.measurements_encode = nn.Sequential(
            *[
                nn.Linear(7, 64),
                nn.Sigmoid(),
                nn.Linear(64, embed_dim)
            ]
        )

        self.rgb_patch_embed = self.rgb_backbone

        self.global_embed = nn.Parameter(torch.zeros(1, embed_dim, 5))
        
        
        self.global_embed = nn.Parameter(torch.zeros(1, embed_dim, 5))
        self.view_embed = nn.Parameter(torch.zeros(1, embed_dim, 5, 1))

        self.waypoint_embed = nn.Parameter(torch.zeros(11, 1, embed_dim))

        if self.waypoints_pred_head == "heatmap":
            self.query_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, 5))
            self.query_embed = nn.Parameter(torch.zeros(400 + 5, 1, embed_dim))
        else:
            self.query_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, 11))
            self.query_embed = nn.Parameter(torch.zeros(400 + 11, 1, embed_dim))

        if self.waypoints_pred_head == "heatmap":
            self.waypoints_generator = MultiPath_Generator(
                embed_dim + 32, embed_dim, 10
            )
        elif self.waypoints_pred_head == "gru":
            self.waypoints_generator = GRUWaypointsPredictor(embed_dim)
        elif self.waypoints_pred_head == "gru-command":
            self.waypoints_generator = GRUWaypointsPredictorWithCommand(embed_dim)
        elif self.waypoints_pred_head == "linear":
            self.waypoints_generator = LinearWaypointsPredictor(embed_dim)
        elif self.waypoints_pred_head == "linear-sum":
            self.waypoints_generator = LinearWaypointsPredictor(embed_dim, cumsum=True)

        self.junction_pred_head = nn.Sequential(
            *[
                    nn.Linear(embed_dim, 64),
                    nn.Dropout(0.5),
                    nn.Linear(64, 7),
                    nn.Sigmoid(),
                    nn.Linear(7, 2)
                ]
            )
        
        self.brake_gap = nn.AdaptiveAvgPool1d(1)
        self.brake_pred_head = nn.Sequential(
            *[
                    nn.Linear(embed_dim , 64),
                    nn.Dropout(0.5),
                    nn.Linear(64, 7),
                    nn.ReLU(),
                    nn.Linear(7, 2)
                ]
            )
        self.traffic_light_pred_head = nn.Sequential(
            *[
                    nn.Linear(embed_dim, 64),
                    nn.Dropout(0.5),
                    nn.Linear(64, 7),
                    nn.ReLU(),
                    nn.Linear(7, 2)
                ]
            )
        
        self.stop_sign_head = nn.Sequential(
            *[
                    nn.Linear(embed_dim, 64),
                    nn.Dropout(0.5),
                    nn.Linear(64, 7),
                    nn.ReLU(),
                    nn.Linear(7, 2)
                ]
            )

        if self.traffic_pred_head_type == "det":
            self.traffic_pred_head = nn.Sequential(
                *[
                    nn.Linear(embed_dim + 32, 64),
                    nn.ReLU(),
                    nn.Linear(64, 7),
                    nn.Sigmoid(),
                ]
            )
        elif self.traffic_pred_head_type == "seg":
            self.traffic_pred_head = nn.Sequential(
                *[nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()]
            )

        self.position_encoding = PositionEmbeddingSine(embed_dim // 2, normalize=True)


        encoder_layer = TransformerEncoderLayer(
            embed_dim, num_heads, dim_feedforward, dropout, act_layer, enc_depth,
            normalize_before
        )
        self.encoder = TransformerEncoder(encoder_layer, enc_depth, None)


        decoder_layer = TransformerDecoderLayer(
            embed_dim, num_heads, dim_feedforward, dropout, act_layer, dec_depth,
            normalize_before
        )
        decoder_norm = nn.LayerNorm(embed_dim)
        self.decoder = TransformerDecoder(
            decoder_layer, dec_depth, decoder_norm, return_intermediate=False
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.global_embed)
        nn.init.uniform_(self.view_embed)
        nn.init.uniform_(self.query_embed)
        nn.init.uniform_(self.query_pos_embed)

        nn.init.uniform_(self.waypoint_embed)


    def forward_features(
        self,
        front_image,
        left_image,
        right_image,
        front_center_image,
        measurements,
    ):
        features = []

        # Front view processing
        front_image_token = self.rgb_patch_embed(front_image)

        if self.use_view_embed:
            front_image_token = (
                front_image_token
                + self.view_embed[:, :, 0:1, :]
                + self.position_encoding(front_image_token)
            )
        else:
            front_image_token = front_image_token + self.position_encoding(
                front_image_token
            )
        

        front_image_token = front_image_token.flatten(2).permute(2, 0, 1)

        features.extend([front_image_token])

        if self.with_right_left_sensors:
            # Left view processing
            left_image_token = self.rgb_patch_embed(left_image)
            if self.use_view_embed:
                left_image_token = (
                    left_image_token
                    + self.view_embed[:, :, 1:2, :]
                    + self.position_encoding(left_image_token)
                )
            else:
                left_image_token = left_image_token + self.position_encoding(
                    left_image_token
                )
            left_image_token = left_image_token.flatten(2).permute(2, 0, 1)

            # Right view processing
            right_image_token = self.rgb_patch_embed(
                right_image
            )
            if self.use_view_embed:
                right_image_token = (
                    right_image_token
                    + self.view_embed[:, :, 2:3, :]
                    + self.position_encoding(right_image_token)
                )
            else:
                right_image_token = right_image_token + self.position_encoding(
                    right_image_token
                )
            right_image_token = right_image_token.flatten(2).permute(2, 0, 1)

            features.extend(
                [
                    left_image_token,
                    right_image_token,
                ]
            )

        if self.with_center_sensor:
            # Front center view processing
            front_center_image_token = self.rgb_patch_embed(front_center_image)
            
            if self.use_view_embed:
                front_center_image_token = (
                    front_center_image_token
                    + self.view_embed[:, :, 3:4, :]
                    + self.position_encoding(front_center_image_token)
                )
            else:
                front_center_image_token = (
                    front_center_image_token
                    + self.position_encoding(front_center_image_token)
                )

            front_center_image_token = front_center_image_token.flatten(2).permute(
                2, 0, 1
            )

            features.extend([front_center_image_token])

        features = torch.cat(features, 0)

        return features
        

    def forward_val(self, x):

        front_image = x["rgb"]
        left_image = x["rgb_left"]
        right_image = x["rgb_right"]
        front_center_image = x["rgb_center"]
        measurements = x["measurements"]
        target_point = x["target_point"]
    
        if self.direct_concat:
            img_size = front_image.shape[-1]
            left_image = torch.nn.functional.interpolate(
                left_image, size=(img_size, img_size)
            )
            right_image = torch.nn.functional.interpolate(
                right_image, size=(img_size, img_size)
            )
            front_center_image = torch.nn.functional.interpolate(
                front_center_image, size=(img_size, img_size)
            )
            front_image = torch.cat(
                [front_image, left_image, right_image, front_center_image], dim=1
            )



        features = self.forward_features(
            front_image,
            left_image,
            right_image,
            front_center_image,
            measurements,
        )

        bs = front_image.shape[0]

        tgt = self.position_encoding(
            torch.ones((bs, 1, 20, 20), device=x["rgb"].device)
        )
        tgt = tgt.flatten(2)
        tgt = torch.cat([tgt, self.query_pos_embed.repeat(bs, 1, 1)], 2)
        tgt = tgt.permute(2, 0, 1)

        meas = self.measurements_encode(measurements).unsqueeze(0)

        features = torch.cat((features, meas), dim=0)

        features = torch.cat((features, self.waypoint_embed.repeat(1, bs, 1)), dim=0)
        memory = self.encoder(features, mask=self.attn_mask)

        memory = memory.permute(1, 0, 2)  # Batchsize ,  N, C

        is_junction_feature = memory[:, 11]
        traffic_light_state_feature = memory[:, 11]
        stop_sign_feature = memory[:, 11]
        brake_feature = torch.cat((memory, features.permute(1, 0, 2)), dim=1).permute(0, 2, 1)
        waypoints_feature = memory[:, 0:10]

        waypoints = self.waypoints_generator(waypoints_feature, target_point)


        is_junction = self.junction_pred_head(is_junction_feature)
        

        traffic_light_state = torch.softmax(self.traffic_light_pred_head(traffic_light_state_feature), dim=1)
        
        stop_sign = self.stop_sign_head(stop_sign_feature)

        
        brake_feature = self.brake_gap(brake_feature).squeeze(2)
        brake = torch.softmax(self.brake_pred_head(brake_feature), dim=1)


        return brake, waypoints, is_junction, traffic_light_state, stop_sign

    def forward(self, x):
        front_image = x["rgb"]
        left_image = x["rgb_left"]
        right_image = x["rgb_right"]
        front_center_image = x["rgb_center"]
        measurements = x["measurements"]
        target_point = x["target_point"]
    
        if self.direct_concat:
            img_size = front_image.shape[-1]
            left_image = torch.nn.functional.interpolate(
                left_image, size=(img_size, img_size)
            )
            right_image = torch.nn.functional.interpolate(
                right_image, size=(img_size, img_size)
            )
            front_center_image = torch.nn.functional.interpolate(
                front_center_image, size=(img_size, img_size)
            )
            front_image = torch.cat(
                [front_image, left_image, right_image, front_center_image], dim=1
            )



        features = self.forward_features(
            front_image,
            left_image,
            right_image,
            front_center_image,
            measurements,
        )

        bs = front_image.shape[0]

        

        tgt = self.position_encoding(
            torch.ones((bs, 1, 20, 20), device=x["rgb"].device)
        )
        tgt = tgt.flatten(2)
        tgt = torch.cat([tgt, self.query_pos_embed.repeat(bs, 1, 1)], 2)
        tgt = tgt.permute(2, 0, 1)

        meas = self.measurements_encode(measurements).unsqueeze(0)

        features = torch.cat((features, meas), dim=0)

        features = torch.cat((features, self.waypoint_embed.repeat(1, bs, 1)), dim=0)
        memory = self.encoder(features, mask=self.attn_mask)
    
        hs = self.decoder(self.query_embed.repeat(1, bs, 1), memory, query_pos=tgt)[0]

        hs = hs.permute(1, 0, 2)  # Batchsize ,  N, C

        traffic_feature = hs[:, :400]
        
        velocity = measurements[:, 6:7].unsqueeze(-1)
        velocity = velocity.repeat(1, 400, 32)
        traffic_feature_with_vel = torch.cat([traffic_feature, velocity], dim=2)
        traffic = self.traffic_pred_head(traffic_feature_with_vel)

        memory = memory.permute(1, 0, 2)  # Batchsize ,  N, C

        is_junction_feature = memory[:, 11]
        traffic_light_state_feature = memory[:, 11]
        stop_sign_feature = memory[:, 11]
        brake_feature = torch.cat((memory, features.permute(1, 0, 2), meas.permute(1, 0, 2)), dim=1).permute(0, 2, 1)
        waypoints_feature = memory[:, 0:10]

        waypoints = self.waypoints_generator(waypoints_feature, target_point)


        is_junction = self.junction_pred_head(is_junction_feature)
        

        traffic_light_state = torch.softmax(self.traffic_light_pred_head(traffic_light_state_feature), dim=1)
        
        stop_sign = self.stop_sign_head(stop_sign_feature)

        # velocity = measurements[:, 6:7].unsqueeze(-1)
        # velocity = self.traffic_pred_head(velocity)
        
        brake_feature = self.brake_gap(brake_feature).squeeze(2)
        brake = torch.softmax(self.brake_pred_head(brake_feature), dim=1)


        return brake, waypoints, is_junction, traffic_light_state, stop_sign, traffic

        # return brake, waypoints, is_junction, traffic_light_state, stop_sign


@register_model
def DemoEnd2EndNet_baseline(**kwargs):
    model = DemoEnd2EndNet(
        enc_depth=8,
        dec_depth=6,
        embed_dim=512,
        rgb_backbone_name="gc_ViT",
        waypoints_pred_head="gru"
    )

    return model

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.nn as nn
from timm.models.layers import StdConv2dSame, StdConv2d, to_2tuple
import math
from typing import Optional, List
import copy
import numpy as np

import warnings

import admin_torch
from timm import create_model

# from gc_vit import GCViTLayer, PatchEmbed, trunc_normal_, _to_channel_first


class HybridEmbed(nn.Module):
    def __init__(
        self,
        backbone,
        img_size=224,
        patch_size=1,
        feature_size=None,
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, "feature_info"):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features

        self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.backbone(x)
        xs = None
        if isinstance(x, (list, tuple)):
            xs = x
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x)
        global_x = torch.mean(x, [2, 3], keepdim=False)[:, :, None]
        return x, global_x, xs

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))



class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)



class YOLOHead(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.l10 = Conv(1024, 512, 1, 1)
        self.l11 = nn.Upsample(None, 2, 'nearest')
        self.l12_P4 = Concat(1)
        self.l13 = nn.Sequential(
            C3(1024, 512, False),
            *[C3(512, 512, False) for _ in range(2)]
            )

        self.conv = Conv(512, 2, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input):

        _, p4_x, x = input
        x = self.l10(x)
        x = self.l11(x)
        p4 = self.l12_P4((x, p4_x))

        x = self.l13(p4)

        x = self.gap(self.conv(x)).squeeze(2).squeeze(2)

        return x

    pass

class YOLOBackbone(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.l0_P1 = Conv(3, 64, 6, 2, 2)
        self.l1_P2 = Conv(64, 128, 3, 2)
        self.l2 = nn.Sequential(*[C3(128, 128) for _ in range(3)])
        self.l3_P3 = Conv(128, 256, 3, 2)
        self.l4 = nn.Sequential(*[C3(256, 256) for _ in range(6)])
        self.l5_P4 = Conv(256, 512, 3, 2)
        self.l6 = nn.Sequential(*[C3(512, 512) for _ in range(9)])
        self.l7_P5 = Conv(512, 1024, 3, 2)
        self.l8 = nn.Sequential(*[C3(1024, 1024) for _ in range(3)])
        self.l9 = SPPF(1024, 1024, 5)


    def forward(self, x):

        p1 = self.l0_P1(x)
        p2 = self.l1_P2(p1)
        x = self.l2(p2)
        p3 = self.l3_P3(x)
        x = self.l4(p3)
        p4 = self.l5_P4(x)
        x = self.l6(p4)
        p5 = self.l7_P5(x)
        x = self.l8(p5)
        x = self.l9(x)

        return (p3, p4, x)

    pass

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        x = tensor
        bs, _, h, w = x.shape
        not_mask = torch.ones((bs, h, w), device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class SpatialSoftmax(nn.Module):
    def __init__(self, height, width, channel, temperature=None, data_format="NCHW"):
        super().__init__()

        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = nn.Parameter(torch.ones(1) * temperature)
        else:
            self.temperature = 1.0

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self.height), np.linspace(-1.0, 1.0, self.width)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...

        if self.data_format == "NHWC":
            feature = (
                feature.transpose(1, 3)
                .tranpose(2, 3)
                .view(-1, self.height * self.width)
            )
        else:
            feature = feature.view(-1, self.height * self.width)

        weight = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(
            torch.autograd.Variable(self.pos_x) * weight, dim=1, keepdim=True
        )
        expected_y = torch.sum(
            torch.autograd.Variable(self.pos_y) * weight, dim=1, keepdim=True
        )
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel, 2)
        feature_keypoints[:, :, 1] = (feature_keypoints[:, :, 1] - 1) * 12
        feature_keypoints[:, :, 0] = feature_keypoints[:, :, 0] * 12
        return feature_keypoints


class MultiPath_Generator(nn.Module):
    def __init__(self, in_channel, embed_dim, out_channel):
        super().__init__()
        self.spatial_softmax = SpatialSoftmax(100, 100, out_channel)
        self.tconv0 = nn.Sequential(
            nn.ConvTranspose2d(in_channel, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 192, 4, 2, 1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.tconv4_list = torch.nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(64, out_channel, 8, 2, 3, bias=False),
                    nn.Tanh(),
                )
                for _ in range(6)
            ]
        )

        self.upsample = nn.Upsample(size=(50, 50), mode="bilinear")

    def forward(self, x, measurements):
        mask = measurements[:, :6]
        mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 100, 100)
        velocity = measurements[:, 6:7].unsqueeze(-1).unsqueeze(-1)
        velocity = velocity.repeat(1, 32, 2, 2)

        n, d, c = x.shape
        x = x.transpose(1, 2)
        x = x.view(n, -1, 2, 2)
        x = torch.cat([x, velocity], dim=1)
        x = self.tconv0(x)
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.upsample(x)
        xs = []
        for i in range(6):
            xt = self.tconv4_list[i](x)
            xs.append(xt)
        xs = torch.stack(xs, dim=1)
        x = torch.sum(xs * mask, dim=1)
        x = self.spatial_softmax(x)
        return x


class LinearWaypointsPredictor(nn.Module):
    def __init__(self, input_dim, cumsum=True):
        super().__init__()
        self.cumsum = cumsum
        self.rank_embed = nn.Parameter(torch.zeros(1, 10, input_dim))
        self.head_fc1_list = nn.ModuleList([nn.Linear(input_dim, 64) for _ in range(6)])
        self.head_relu = nn.ReLU(inplace=True)
        self.head_fc2_list = nn.ModuleList([nn.Linear(64, 2) for _ in range(6)])

    def forward(self, x, measurements):
        # input shape: n 10 embed_dim
        bs, n, dim = x.shape
        x = x + self.rank_embed
        x = x.reshape(-1, dim)

        mask = measurements[:, :6]
        mask = torch.unsqueeze(mask, -1).repeat(n, 1, 2)

        rs = []
        for i in range(6):
            res = self.head_fc1_list[i](x)
            res = self.head_relu(res)
            res = self.head_fc2_list[i](res)
            rs.append(res)
        rs = torch.stack(rs, 1)
        x = torch.sum(rs * mask, dim=1)

        x = x.view(bs, n, 2)
        if self.cumsum:
            x = torch.cumsum(x, 1)
        return x


class GRUWaypointsPredictor(nn.Module):
    def __init__(self, input_dim, waypoints=10):
        super().__init__()
        # self.gru = torch.nn.GRUCell(input_size=input_dim, hidden_size=64)
        self.gru = torch.nn.GRU(input_size=input_dim, hidden_size=64, batch_first=True)
        self.encoder = nn.Linear(2, 64)
        self.decoder = nn.Linear(64, 2)
        self.waypoints = waypoints

    def forward(self, x, target_point):
        bs = x.shape[0]
        z = self.encoder(target_point).unsqueeze(0)
        output, _ = self.gru(x, z)
        output = output.reshape(bs * self.waypoints, -1)
        output = self.decoder(output).reshape(bs, self.waypoints, 2)
        output = torch.cumsum(output, 1)
        return output

class GRUWaypointsPredictorWithCommand(nn.Module):
    def __init__(self, input_dim, waypoints=10):
        super().__init__()
        # self.gru = torch.nn.GRUCell(input_size=input_dim, hidden_size=64)
        self.grus = nn.ModuleList([torch.nn.GRU(input_size=input_dim, hidden_size=64, batch_first=True) for _ in range(6)])
        self.encoder = nn.Linear(2, 64)
        self.decoders = nn.ModuleList([nn.Linear(64, 2) for _ in range(6)])
        self.waypoints = waypoints

    def forward(self, x, target_point, measurements):
        bs, n, dim = x.shape
        mask = measurements[:, :6, None, None]
        mask = mask.repeat(1, 1, self.waypoints, 2)

        z = self.encoder(target_point).unsqueeze(0)
        outputs = []
        for i in range(6):
            output, _ = self.grus[i](x, z)
            output = output.reshape(bs * self.waypoints, -1)
            output = self.decoders[i](output).reshape(bs, self.waypoints, 2)
            output = torch.cumsum(output, 1)
            outputs.append(output)
        outputs = torch.stack(outputs, 1)
        output = torch.sum(outputs * mask, dim=1)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=nn.ReLU(),
        encoder_layers=6,
        normalize_before=False,
        use_admin=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation()
        self.normalize_before = normalize_before

        self.use_admin = use_admin

        if self.use_admin:
            
            num_layer =  2 * encoder_layers
            self.residual_attn = admin_torch.as_module(num_layer)

            self.residual_ffn = admin_torch.as_module(num_layer)
            pass
        

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        

        if self.use_admin:
            f_x, _ = self.self_attn(
                q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
                )
            src = self.residual_attn(src, self.dropout1(f_x))
            src = self.norm1(src)

            src2 = self.linear2(self.activation(self.linear1(src)))
            
            self.residual_ffn(src, self.dropout2(src2))
            
            src = self.norm2(src)

        else:
            src2 = self.self_attn(
                    q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
                )[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)

            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class GPTLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=nn.ReLU(),
        decoder_layers=6,
        normalize_before=False,
        use_admin=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation()
        self.normalize_before = normalize_before

        self.use_admin = use_admin
        if self.use_admin:
            
            num_layer =  2 * decoder_layers
            self.residual_attn = admin_torch.as_module(num_layer)
            self.residual_attn2 = admin_torch.as_module(num_layer)
            self.residual_ffn = admin_torch.as_module(num_layer)
            pass

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        # memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]

        if self.use_admin:
            tgt = self.residual_attn(tgt, self.dropout1(tgt2))
            tgt = self.norm1(tgt)

            # tgt2 = self.multihead_attn(
            #     query=self.with_pos_embed(tgt, query_pos),
            #     key=self.with_pos_embed(memory, pos),
            #     value=memory,
            #     attn_mask=memory_mask,
            #     key_padding_mask=memory_key_padding_mask,
            # )[0]

            tgt = self.residual_attn2(tgt, self.dropout2(tgt2))
            tgt = self.norm2(tgt)

            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = self.residual_ffn(tgt, self.dropout3(tgt2))
            tgt = self.norm3(tgt)
        else:
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            # tgt2 = self.multihead_attn(
            #     query=self.with_pos_embed(tgt, query_pos),
            #     key=self.with_pos_embed(memory, pos),
            #     value=memory,
            #     attn_mask=memory_mask,
            #     key_padding_mask=memory_key_padding_mask,
            # )[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):

        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=nn.ReLU(),
        decoder_layers=6,
        normalize_before=False,
        use_admin=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation()
        self.normalize_before = normalize_before

        self.use_admin = use_admin
        if self.use_admin:
            
            num_layer =  2 * decoder_layers
            self.residual_attn = admin_torch.as_module(num_layer)
            self.residual_attn2 = admin_torch.as_module(num_layer)
            self.residual_ffn = admin_torch.as_module(num_layer)
            pass

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]

        if self.use_admin:
            tgt = self.residual_attn(tgt, self.dropout1(tgt2))
            tgt = self.norm1(tgt)

            tgt2 = self.multihead_attn(
                query=self.with_pos_embed(tgt, query_pos),
                key=self.with_pos_embed(memory, pos),
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )[0]

            tgt = self.residual_attn2(tgt, self.dropout2(tgt2))
            tgt = self.norm2(tgt)

            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = self.residual_ffn(tgt, self.dropout3(tgt2))
            tgt = self.norm3(tgt)
        else:
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            tgt2 = self.multihead_attn(
                query=self.with_pos_embed(tgt, query_pos),
                key=self.with_pos_embed(memory, pos),
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )

import os

class GC_ViTBackBone(nn.Module):

    def __init__(self, ) -> None:
        super().__init__()

        self.model = create_model('gc_vit_xtiny', pretrained=False)
        project_dir = os.environ['project_dir']
        self.model.load_state_dict(torch.load("{}/DemoEnd2EndNet/gcvit_1k_xtiny.pth.tar".format(project_dir))['state_dict'])
    

    def forward(self, x):
        x = self.model.patch_embed(x)
        x = self.model.pos_drop(x)

        for level in self.model.levels:
            x = level(x)

        x = self.model.norm(x)
        x = x.permute(0, 3, 1, 2)
        # x = self.model.avgpool(x)

        return x
    pass
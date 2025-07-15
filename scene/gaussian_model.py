#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import time
from functools import reduce

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_max

from utils.general_utils import (build_scaling_rotation, get_expon_lr_func,
                                 inverse_sigmoid, strip_symmetric)
from utils.graphics_utils import BasicPointCloud
from utils.system_utils import mkdir_p
from utils.entropy_models import Entropy_bernoulli, Entropy_gaussian, Entropy_factorized, Entropy_gaussian_mix_prob_2, Entropy_gaussian_mix_prob_3

from utils.encodings import \
    STE_binary, STE_multistep, Quantize_anchor, \
    anchor_round_digits, \
    get_binary_vxl_size, \
    GridEncoder

from utils.encodings_cuda import \
    encoder, decoder, \
    encoder_gaussian_chunk, decoder_gaussian_chunk, encoder_gaussian_mixed_chunk, decoder_gaussian_mixed_chunk
from utils.gpcc_utils import compress_gpcc, decompress_gpcc, calculate_morton_order
import MinkowskiEngine as ME
import faiss
from einops import rearrange
import math
from sklearn.neighbors import LocalOutlierFactor
from compressai.ops import quantize_ste

bit2MB_scale = 8 * 1024 * 1024
MAX_batch_size = 3000

def get_time():
    torch.cuda.synchronize()
    tt = time.time()
    return tt

class mix_3D2D_encoding(nn.Module):
    def __init__(
            self,
            n_features,
            resolutions_list,
            log2_hashmap_size,
            resolutions_list_2D,
            log2_hashmap_size_2D,
            ste_binary,
            ste_multistep,
            add_noise,
            Q,
    ):
        super().__init__()
        self.encoding_xyz = GridEncoder(
            num_dim=3,
            n_features=n_features,
            resolutions_list=resolutions_list,
            log2_hashmap_size=log2_hashmap_size,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.encoding_xy = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.encoding_xz = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.encoding_yz = GridEncoder(
            num_dim=2,
            n_features=n_features,
            resolutions_list=resolutions_list_2D,
            log2_hashmap_size=log2_hashmap_size_2D,
            ste_binary=ste_binary,
            ste_multistep=ste_multistep,
            add_noise=add_noise,
            Q=Q,
        )
        self.output_dim = self.encoding_xyz.output_dim + \
                          self.encoding_xy.output_dim + \
                          self.encoding_xz.output_dim + \
                          self.encoding_yz.output_dim

    def forward(self, x):
        x_x, y_y, z_z = torch.chunk(x, 3, dim=-1)
        out_xyz = self.encoding_xyz(x)  # [..., 2*16]
        out_xy = self.encoding_xy(torch.cat([x_x, y_y], dim=-1))  # [..., 2*4]
        out_xz = self.encoding_xz(torch.cat([x_x, z_z], dim=-1))  # [..., 2*4]
        out_yz = self.encoding_yz(torch.cat([y_y, z_z], dim=-1))  # [..., 2*4]
        out_i = torch.cat([out_xyz, out_xy, out_xz, out_yz], dim=-1)  # [..., 56]
        return out_i

class Channel_CTX_fea(nn.Module):
    def __init__(self):
        super().__init__()
        self.MLP_d0 = nn.Sequential(
            nn.Linear(50*3+0, 20*2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20*2, 5*3),
        )
        self.MLP_d1 = nn.Sequential(
            nn.Linear(50*3+5, 20*2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20*2, 10*3),
        )
        self.MLP_d2 = nn.Sequential(
            nn.Linear(50*3+5+10, 20*2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20*2, 15*3),
        )
        self.MLP_d3 = nn.Sequential(
            nn.Linear(50*3+5+10+15, 20*2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20*2, 20*3),
        )

    def forward(self, fea_q, mean_scale, to_dec=-1):  # chctx_v3
        if to_dec == -1:
            # fea_q: [N, 50]
            d0, d1, d2, d3 = torch.split(fea_q, split_size_or_sections=[5, 10, 15, 20], dim=-1)
            mean_d0, scale_d0, prob_d0 = torch.chunk(self.MLP_d0(torch.cat([mean_scale], dim=-1)), chunks=3, dim=-1)
            mean_d1, scale_d1, prob_d1 = torch.chunk(self.MLP_d1(torch.cat([d0, mean_scale], dim=-1)), chunks=3, dim=-1)
            mean_d2, scale_d2, prob_d2 = torch.chunk(self.MLP_d2(torch.cat([d0, d1, mean_scale], dim=-1)), chunks=3, dim=-1)
            mean_d3, scale_d3, prob_d3 = torch.chunk(self.MLP_d3(torch.cat([d0, d1, d2, mean_scale], dim=-1)), chunks=3, dim=-1)

            mean_adj = torch.cat([mean_d0, mean_d1, mean_d2, mean_d3], dim=-1)
            scale_adj = torch.cat([scale_d0, scale_d1, scale_d2, scale_d3], dim=-1)
            prob_adj = torch.cat([prob_d0, prob_d1, prob_d2, prob_d3], dim=-1)
        
            return mean_adj, scale_adj, prob_adj
        
        else :
            MLP = getattr(self, f"MLP_d{to_dec}")
            mean_adj, scale_adj, prob_adj = torch.chunk(MLP(torch.cat([fea_q[..., :5*to_dec*(to_dec+1)/2], mean_scale], dim=-1)), chunks=3, dim=-1)

            return mean_adj, scale_adj, prob_adj
        

class Channel_CTX_fea_medium(nn.Module):
    def __init__(self):
        super().__init__()
        self.MLP_d0 = nn.Sequential(
            nn.Linear(5*3+0, 20*2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20*2, 5*3),
        )
        self.MLP_d1 = nn.Sequential(
            nn.Linear(10*3+5, 20*2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20*2, 10*3),
        )
        self.MLP_d2 = nn.Sequential(
            nn.Linear(15*3+5+10, 20*2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20*2, 15*3),
        )
        self.MLP_d3 = nn.Sequential(
            nn.Linear(20*3+5+10+15, 20*2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(20*2, 20*3),
        )

    def forward(self, fea_q, mean_scale, to_dec=-1):  # chctx_v3
        # fea_q: [N, 50]
        if to_dec == -1:
            d0, d1, d2, d3 = torch.split(fea_q, split_size_or_sections=[5, 10, 15, 20], dim=-1)
            mean_scale_0, mean_scale_1, mean_scale_2, mean_scale_3 = torch.split(mean_scale.view(-1, 3, 50), split_size_or_sections=[5, 10, 15, 20], dim=-1)
            mean_d0, scale_d0, prob_d0 = torch.chunk(self.MLP_d0(torch.cat([mean_scale_0.reshape(-1, 5*3)], dim=-1)), chunks=3, dim=-1)
            mean_d1, scale_d1, prob_d1 = torch.chunk(self.MLP_d1(torch.cat([d0, mean_scale_1.reshape(-1, 10*3)], dim=-1)), chunks=3, dim=-1)
            mean_d2, scale_d2, prob_d2 = torch.chunk(self.MLP_d2(torch.cat([d0, d1, mean_scale_2.reshape(-1, 15*3)], dim=-1)), chunks=3, dim=-1)
            mean_d3, scale_d3, prob_d3 = torch.chunk(self.MLP_d3(torch.cat([d0, d1, d2, mean_scale_3.reshape(-1, 20*3)], dim=-1)), chunks=3, dim=-1)

            mean_adj = torch.cat([mean_d0, mean_d1, mean_d2, mean_d3], dim=-1)
            scale_adj = torch.cat([scale_d0, scale_d1, scale_d2, scale_d3], dim=-1)
            prob_adj = torch.cat([prob_d0, prob_d1, prob_d2, prob_d3], dim=-1)

            return mean_adj, scale_adj, prob_adj
        
        else:
            MLP = getattr(self, f"MLP_d{to_dec}")
            start = 5*to_dec*(to_dec+1)/2
            end = 5*(to_dec+2)*(to_dec+1)/2
            mean_scale = mean_scale.view(-1, 3, 50)[..., start:end].reshape(-1, 10*3)
            mean_adj, scale_adj, prob_adj = torch.chunk(MLP(torch.cat([fea_q[..., :5*to_dec*(to_dec+1)/2], mean_scale], dim=-1)), chunks=3, dim=-1)
            
            return mean_adj, scale_adj, prob_adj

class Channel_CTX_fea_tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_d0 = nn.Parameter(torch.zeros(size=[1, 5]))
        self.scale_d0 = nn.Parameter(torch.zeros(size=[1, 5]))
        self.prob_d0 = nn.Parameter(torch.zeros(size=[1, 5]))
        self.MLP_d1 = nn.Sequential(
            nn.Linear(5, 10*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*3, 10*3),
        )
        self.MLP_d2 = nn.Sequential(
            nn.Linear(5+10, 10*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*3, 15*3),
        )
        self.MLP_d3 = nn.Sequential(
            nn.Linear(5+10+15, 10*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*3, 20*3),
        )

    def forward(self, fea_q, to_dec=-1):  # chctx_v3
        # fea_q: [N, 50]
        if to_dec == -1:
            NN = fea_q.shape[0]
            d0, d1, d2, d3 = torch.split(fea_q, split_size_or_sections=[5, 10, 15, 20], dim=-1)
            mean_d0, scale_d0, prob_d0 = self.mean_d0.repeat(NN, 1), self.scale_d0.repeat(NN, 1), self.prob_d0.repeat(NN, 1)
            mean_d1, scale_d1, prob_d1 = torch.chunk(self.MLP_d1(torch.cat([d0], dim=-1)), chunks=3, dim=-1)
            mean_d2, scale_d2, prob_d2 = torch.chunk(self.MLP_d2(torch.cat([d0, d1], dim=-1)), chunks=3, dim=-1)
            mean_d3, scale_d3, prob_d3 = torch.chunk(self.MLP_d3(torch.cat([d0, d1, d2], dim=-1)), chunks=3, dim=-1)

            mean_adj = torch.cat([mean_d0, mean_d1, mean_d2, mean_d3], dim=-1)
            scale_adj = torch.cat([scale_d0, scale_d1, scale_d2, scale_d3], dim=-1)
            prob_adj = torch.cat([prob_d0, prob_d1, prob_d2, prob_d3], dim=-1)

            return mean_adj, scale_adj, prob_adj

        elif to_dec == 0:
            NN = fea_q.shape[0]
            mean_d0, scale_d0, prob_d0 = self.mean_d0.repeat(NN, 1), self.scale_d0.repeat(NN, 1), self.prob_d0.repeat(NN, 1)

            return mean_d0, scale_d0, prob_d0

        else:
            MLP = getattr(self, f"MLP_d{to_dec}")
            mean_adj, scale_adj, prob_adj = torch.chunk(MLP(fea_q[..., :5*to_dec*(to_dec+1)/2]), chunks=3, dim=-1)

            return mean_adj, scale_adj, prob_adj


class Spatial_CTX_fea_Minko(ME.MinkowskiNetwork):
    def __init__(self, D=3, mask_type: str = "A",):
        super(Spatial_CTX_fea_Minko, self).__init__(D)
        
        self.MLP_d0 = nn.Sequential(
            ME.MinkowskiConvolution(10, 20*2, kernel_size=3, stride=1, bias=False, dimension=3),
            ME.MinkowskiLeakyReLU(inplace=True),
            ME.MinkowskiConvolution(20*2, 10*3, kernel_size=1, stride=1, bias=False, dimension=3),
        ).cuda()

        # self.mask = torch.ones_like(self.MLP_d0[0].weight.data)
        # _, _, d, h, w = self.mask.size()
        # self.mask[:, :, 0::2, 0::2, 1::2] = 0
        # self.mask[:, :, 0::2, 1::2, 0::2] = 0
        # self.mask[:, :, 1::2, 0::2, 0::2] = 0
        # self.mask[:, :, 1::2, 1::2, 1::2] = 0
        # self.mask[:, :, d//2, h // 2, w // 2] = mask_type == "B"

    def forward(self, fea_q, coords, anchor_sign, to_dec=-1):  # chctx_v3
        # fea_q: [N, 50]
        fea_q[~anchor_sign, ...] = 0
        d0, d1, d2, d3, d4 = torch.split(fea_q, split_size_or_sections=[10, 10, 10, 10, 10], dim=-1)
        N = coords.shape[0]
        batch = torch.zeros(N, 1).cuda().int()
        coords = torch.cat((batch, coords), dim=1) 
        # coord_manager = ME.CoordinateManager(D=3)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sp_0 = ME.SparseTensor(features=d0, coordinates=coords, device=device)
        sp_1 = ME.SparseTensor(features=d1, coordinate_map_key=sp_0.coordinate_map_key, coordinate_manager=sp_0.coordinate_manager, device=device)
        sp_2 = ME.SparseTensor(features=d2, coordinate_map_key=sp_0.coordinate_map_key, coordinate_manager=sp_0.coordinate_manager, device=device)
        sp_3 = ME.SparseTensor(features=d3, coordinate_map_key=sp_0.coordinate_map_key, coordinate_manager=sp_0.coordinate_manager, device=device)
        sp_4 = ME.SparseTensor(features=d4, coordinate_map_key=sp_0.coordinate_map_key, coordinate_manager=sp_0.coordinate_manager, device=device)


        mean_d0, scale_d0, prob_d0 = torch.chunk(self.MLP_d0(sp_0).F, chunks=3, dim=-1)
        mean_d1, scale_d1, prob_d1 = torch.chunk(self.MLP_d0(sp_1).F, chunks=3, dim=-1)
        mean_d2, scale_d2, prob_d2 = torch.chunk(self.MLP_d0(sp_2).F, chunks=3, dim=-1)
        mean_d3, scale_d3, prob_d3 = torch.chunk(self.MLP_d0(sp_3).F, chunks=3, dim=-1)
        mean_d4, scale_d4, prob_d4 = torch.chunk(self.MLP_d0(sp_4).F, chunks=3, dim=-1)
        mean_sp = torch.cat([mean_d0, mean_d1, mean_d2, mean_d3, mean_d4], dim=-1)
        scale_sp = torch.cat([scale_d0, scale_d1, scale_d2, scale_d3, scale_d4], dim=-1)
        prob_sp = torch.cat([prob_d0, prob_d1, prob_d2, prob_d3, prob_d4], dim=-1)

        if to_dec == 0:
            return mean_d0, scale_d0, prob_d0
        if to_dec == 1:
            return mean_d1, scale_d1, prob_d1
        if to_dec == 2:
            return mean_d2, scale_d2, prob_d2
        if to_dec == 3:
            return mean_d3, scale_d3, prob_d3
        if to_dec == 4:
            return mean_d4, scale_d4, prob_d4
        return mean_sp, scale_sp, prob_sp
    
class Spatial_CTX_fea_knn(nn.Module):
    def __init__(self):
        super().__init__()

        self.MLP_d0 = nn.Sequential(
            nn.Linear(5*3+50*3, 10*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*3, 5*3),
        )
        self.MLP_d1 = nn.Sequential(
            nn.Linear(10*3+50*3, 10*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*3, 10*3),
        )
        self.MLP_d2 = nn.Sequential(
            nn.Linear(15*3+50*3, 10*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*3, 15*3),
        )
        self.MLP_d3 = nn.Sequential(
            nn.Linear(20*3+50*3, 10*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*3, 20*3),
        )

    def forward(self, fea_q, mean_scale, to_dec=-1):
        # fea_q: (nonanchor_num, knn, fea_num)
        d0, d1, d2, d3 = torch.split(fea_q, split_size_or_sections=[5, 10, 15, 20], dim=-1)

        mean_d0, scale_d0, prob_d0 = torch.chunk(self.MLP_d0(torch.cat([d0.reshape(-1, 5*3), mean_scale], dim=-1)), chunks=3, dim=-1)
        mean_d1, scale_d1, prob_d1 = torch.chunk(self.MLP_d1(torch.cat([d1.reshape(-1, 10*3), mean_scale], dim=-1)), chunks=3, dim=-1)
        mean_d2, scale_d2, prob_d2 = torch.chunk(self.MLP_d2(torch.cat([d2.reshape(-1, 15*3), mean_scale], dim=-1)), chunks=3, dim=-1)
        mean_d3, scale_d3, prob_d3 = torch.chunk(self.MLP_d3(torch.cat([d3.reshape(-1, 20*3), mean_scale], dim=-1)), chunks=3, dim=-1)  
        # mean_scale_0, mean_scale_1, mean_scale_2, mean_scale_3 = torch.split(mean_scale.view(-1, 3, 50), split_size_or_sections=[5, 10, 15, 20], dim=-1)
        # mean_d0, scale_d0, prob_d0 = torch.chunk(self.MLP_d0(torch.cat([d0.reshape(-1, 5*5), mean_scale_0.reshape(-1, 5*3)], dim=-1)), chunks=3, dim=-1)
        # mean_d1, scale_d1, prob_d1 = torch.chunk(self.MLP_d1(torch.cat([d1.reshape(-1, 10*5), mean_scale_1.reshape(-1, 10*3)], dim=-1)), chunks=3, dim=-1)
        # mean_d2, scale_d2, prob_d2 = torch.chunk(self.MLP_d2(torch.cat([d2.reshape(-1, 15*5), mean_scale_2.reshape(-1, 15*3)], dim=-1)), chunks=3, dim=-1)
        # mean_d3, scale_d3, prob_d3 = torch.chunk(self.MLP_d3(torch.cat([d3.reshape(-1, 20*5), mean_scale_3.reshape(-1, 20*3)], dim=-1)), chunks=3, dim=-1)
        
        if to_dec == 0:
            return mean_d0, scale_d0, prob_d0
        if to_dec == 1:
            return mean_d1, scale_d1, prob_d1
        if to_dec == 2:
            return mean_d2, scale_d2, prob_d2
        if to_dec == 3:
            return mean_d3, scale_d3, prob_d3
        
        mean_sp = torch.cat([mean_d0, mean_d1, mean_d2, mean_d3], dim=-1)
        scale_sp = torch.cat([scale_d0, scale_d1, scale_d2, scale_d3], dim=-1)
        prob_sp = torch.cat([prob_d0, prob_d1, prob_d2, prob_d3], dim=-1)

        return mean_sp, scale_sp, prob_sp
    
class Spatial_CTX_fea_knn_tiny(nn.Module):
    def __init__(self):
        super().__init__()

        self.MLP_d0 = nn.Sequential(
            nn.Linear(5*3, 10*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*3, 5*3),
        )
        self.MLP_d1 = nn.Sequential(
            nn.Linear(10*3, 10*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*3, 10*3),
        )
        self.MLP_d2 = nn.Sequential(
            nn.Linear(15*3, 10*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*3, 15*3),
        )
        self.MLP_d3 = nn.Sequential(
            nn.Linear(20*3, 10*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10*3, 20*3),
        )

    def forward(self, feat_q, to_dec=-1):
        # fea_q: (nonanchor_num, knn, fea_num)
        d0, d1, d2, d3 = torch.split(feat_q, split_size_or_sections=[5, 10, 15, 20], dim=-1)
        
        mean_d0, scale_d0, prob_d0 = torch.chunk(self.MLP_d0(d0.reshape(-1, 5*3)), chunks=3, dim=-1)
        mean_d1, scale_d1, prob_d1 = torch.chunk(self.MLP_d1(d1.reshape(-1, 10*3)), chunks=3, dim=-1)
        mean_d2, scale_d2, prob_d2 = torch.chunk(self.MLP_d2(d2.reshape(-1, 15*3)), chunks=3, dim=-1)
        mean_d3, scale_d3, prob_d3 = torch.chunk(self.MLP_d3(d3.reshape(-1, 20*3)), chunks=3, dim=-1)  
        # mean_scale_0, mean_scale_1, mean_scale_2, mean_scale_3 = torch.split(mean_scale.view(-1, 3, 50), split_size_or_sections=[5, 10, 15, 20], dim=-1)
        # mean_d0, scale_d0, prob_d0 = torch.chunk(self.MLP_d0(torch.cat([d0.reshape(-1, 5*5), mean_scale_0.reshape(-1, 5*3)], dim=-1)), chunks=3, dim=-1)
        # mean_d1, scale_d1, prob_d1 = torch.chunk(self.MLP_d1(torch.cat([d1.reshape(-1, 10*5), mean_scale_1.reshape(-1, 10*3)], dim=-1)), chunks=3, dim=-1)
        # mean_d2, scale_d2, prob_d2 = torch.chunk(self.MLP_d2(torch.cat([d2.reshape(-1, 15*5), mean_scale_2.reshape(-1, 15*3)], dim=-1)), chunks=3, dim=-1)
        # mean_d3, scale_d3, prob_d3 = torch.chunk(self.MLP_d3(torch.cat([d3.reshape(-1, 20*5), mean_scale_3.reshape(-1, 20*3)], dim=-1)), chunks=3, dim=-1)
        
        if to_dec == 0:
            return mean_d0, scale_d0, prob_d0
        if to_dec == 1:
            return mean_d1, scale_d1, prob_d1
        if to_dec == 2:
            return mean_d2, scale_d2, prob_d2
        if to_dec == 3:
            return mean_d3, scale_d3, prob_d3
        
        mean_sp = torch.cat([mean_d0, mean_d1, mean_d2, mean_d3], dim=-1)
        scale_sp = torch.cat([scale_d0, scale_d1, scale_d2, scale_d3], dim=-1)
        prob_sp = torch.cat([prob_d0, prob_d1, prob_d2, prob_d3], dim=-1)

        return mean_sp, scale_sp, prob_sp


# class Channel_CTX(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.ch_ctx0 = nn.Parameter(torch.zeros(size=[1, 5*2]))
#         self.MLP_d1 = nn.Sequential(
#             nn.Linear(5, 10*3),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(10*3, 10*2),
#         )
#         self.MLP_d2 = nn.Sequential(
#             nn.Linear(15, 15*3),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(15*3, 15*2),
#         )
#         self.MLP_d3 = nn.Sequential(
#             nn.Linear(30, 20*3),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(20*3, 20*2),
#         )

#     def forward(self, fea_q, to_dec=-1):  # chctx_v3
#         # fea_q: [N, 50]
#         if to_dec == -1:
#             NN = fea_q.shape[0]
#             ch_ctx0 = self.ch_ctx0.repeat(NN, 1)
#             ch_ctx1 = self.MLP_d1(fea_q[..., :5])
#             ch_ctx2 = self.MLP_d2(fea_q[..., :15])
#             ch_ctx3 = self.MLP_d3(fea_q[..., :30])

#             ch_ctx = torch.cat([ch_ctx0, ch_ctx1, ch_ctx2, ch_ctx3], dim=-1)

#             return ch_ctx

#         elif to_dec == 0:
#             NN = fea_q.shape[0]
#             ch_ctx0 = self.ch_ctx0.repeat(NN, 1)

#             return ch_ctx0

#         else:
#             MLP = getattr(self, f"MLP_d{to_dec}")
#             ch_ctx = MLP(fea_q[..., :5*to_dec*(to_dec+1)/2])

#             return ch_ctx

class Channel_CTX(nn.Module):
    def __init__(self, num_groups=3, dims=[5, 10, 15, 20]):
        super().__init__()

        self.num_groups = num_groups
        self.dims = dims  # 每层的维度

        self.ch_ctx0 = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 5 * 2)) for _ in range(num_groups)
        ])

        self.groups = nn.ModuleList()

        for g in range(num_groups):
            level_mlps = nn.ModuleList()
            pre_dim = 0
            for d, dim in enumerate(dims):
                if d == 0:
                    pre_dim = dim
                    continue
                input_dim = pre_dim
                hidden_dim = dim * 3
                output_dim = dim * 2
                mlp = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(hidden_dim, output_dim),
                )
                level_mlps.append(mlp)
                pre_dim += dim
            self.groups.append(level_mlps)


    def forward(self, feat_q=None, mask_list=[], choose_idx=None, ch_to_dec=-1, decoding=False, group_to_dec=-1, group_num=0):  # chctx_v3
        if feat_q is not None:
            B, _ = feat_q.shape
            device = feat_q.device
            dtype = feat_q.dtype

        # fea_q: [N, 50]
        if ch_to_dec == -1:
            # if choose_idx is None:
            #     choose_idx = torch.ones(mask_list[0].shape[0], device=device, dtype=torch.bool)
            outputs = torch.zeros(B, sum([d * 2 for d in self.dims]), device=device, dtype=dtype)
            for group_id, mask in enumerate(mask_list):
                if choose_idx is not None:
                    mask = mask[choose_idx]  # shape: (B,)
                group_num = mask.sum()
                if group_num == 0:
                    continue
                group_feat = feat_q[mask]
                pre_dim = 0
                group_out = []

                for i, d in enumerate(self.dims):
                    if i==0:
                        out = self.ch_ctx0[group_id].repeat(group_num, 1)
                        pre_dim=d
                    else:
                        out = self.groups[group_id][i-1](group_feat[..., :pre_dim])   
                        pre_dim += d                   
                    group_out.append(out)

                output_cat = torch.cat(group_out, dim=-1)                     # (M, total_out_dim)
                outputs[mask] = output_cat              # 回填到输出                

            return outputs

        elif decoding and group_to_dec!=-1 and group_num!=0:
            if ch_to_dec == 0:
                outputs = self.ch_ctx0[group_to_dec].repeat(group_num, 1)

            else:
                outputs = self.groups[group_to_dec][ch_to_dec-1](feat_q)  

            return outputs
        else:
            raise ValueError("Wrong input in Channel_CTX")     

# class Spatial_CTX(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.MLP_d0 = nn.Linear(5*2, 5*2)

#         self.MLP_d1 = nn.Linear(10*2, 10*2)

#         self.MLP_d2 = nn.Linear(15*2, 15*2)

#         self.MLP_d3 = nn.Linear(20*2, 20*2)

#     def forward(self, fea_q, to_dec=-1):
#         # fea_q: (nonanchor_num, knn, fea_num)
#         d0, d1, d2, d3 = torch.split(fea_q, split_size_or_sections=[5, 10, 15, 20], dim=-1)
        
#         sp_ctx0 = self.MLP_d0(d0.reshape(-1, 5*2))
#         sp_ctx1 = self.MLP_d1(d1.reshape(-1, 10*2))
#         sp_ctx2 = self.MLP_d2(d2.reshape(-1, 15*2))
#         sp_ctx3 = self.MLP_d3(d3.reshape(-1, 20*2))
        
#         if to_dec == 0:
#             return sp_ctx0
#         if to_dec == 1:
#             return sp_ctx1
#         if to_dec == 2:
#             return sp_ctx2
#         if to_dec == 3:
#             return sp_ctx3
        
#         sp_ctx = torch.cat([sp_ctx0, sp_ctx1, sp_ctx2, sp_ctx3], dim=-1)

#         return sp_ctx

class Spatial_CTX(nn.Module):
    def __init__(self, num_groups=3, dims=[5, 10, 15, 20], knn=2):
        super().__init__()

        self.num_groups = num_groups
        self.dims = dims  # 每层的维度
        self.knn = knn

        # 每个 group 有四层 MLP（对应 d0-d3），共 num_groups 组
        self.groups = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(d * knn, d * 2) for d in dims
            ])
            for _ in range(num_groups-1)
        ])


    def forward(self, feat_q=None, mask_list=[], choose_idx=None, ch_to_dec=-1, decoding=False, group_to_dec=-1):
        # fea_q: (nonanchor_num, knn, fea_num)
        # fea_q shape: (B, knn, sum(dims)) e.g. (N, K, 50)
        if feat_q is not None:
            B, K, _ = feat_q.shape
            device = feat_q.device
            dtype = feat_q.dtype

            

        if ch_to_dec == -1:
            # if choose_idx is None:
            #     choose_idx = torch.ones(mask_list[0].shape[0], device=device, dtype=torch.bool)
            outputs = torch.zeros(B, sum([d * 2 for d in self.dims]), device=device, dtype=dtype)
            split_feats = torch.split(feat_q, self.dims, dim=-1)  # 得到 d0, d1, d2, d3 # (B, K, dim)
            for group_id, mask in enumerate(mask_list):
                if group_id==0:
                    continue
                if choose_idx is not None:
                    mask = mask[choose_idx]  # shape: (B,)
                if mask.sum() == 0:
                    continue

                group_out = []

                for i, d in enumerate(self.dims):
                    feat = split_feats[i][mask]  # (M, K, d)
                    out = self.groups[group_id-1][i](feat.reshape(-1, d * self.knn))                   # (M, d*k)
                    group_out.append(out)

                output_cat = torch.cat(group_out, dim=-1)                     # (M, total_out_dim)
                outputs[mask] = output_cat              # 回填到输出

            return outputs  # shape: (B, total_out_dim)

        elif decoding and group_to_dec!=-1:
            d = self.dims[ch_to_dec]      
                          
            outputs = self.groups[group_to_dec-1][ch_to_dec](feat_q.reshape(-1, d * self.knn))    
            return outputs  # shape: (B*K, d*2)
        else:
            raise ValueError("Wrong input in Spatial_CTX")
        
# class Param_Aggregation(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.MLP_d0 = nn.Sequential(
#             nn.Linear(96+5*4, 5*3),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(5*3, 5*2),
#         )

#         self.MLP_d1 = nn.Sequential(
#             nn.Linear(96+10*4, 10*3),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(10*3, 10*2),
#         )
#         self.MLP_d2 = nn.Sequential(
#             nn.Linear(96+15*4, 15*3),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(15*3, 15*2),
#         )
#         self.MLP_d3 = nn.Sequential(
#             nn.Linear(96+20*4, 20*3),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(20*3, 20*2),
#         )

#     def forward(self, hyper_prior, ch_ctx, sp_ctx, to_dec=-1):
#         # fea_q: (nonanchor_num, knn, fea_num)
#         if to_dec == -1:
#             # hyper0, hyper1, hyper2, hyper3 = torch.split(hyper_prior, split_size_or_sections=[5, 10, 15, 20], dim=-1)
#             ch_ctx0, ch_ctx1, ch_ctx2, ch_ctx3 = torch.split(ch_ctx, split_size_or_sections=[5*2, 10*2, 15*2, 20*2], dim=-1)
#             sp_ctx0, sp_ctx1, sp_ctx2, sp_ctx3 = torch.split(sp_ctx, split_size_or_sections=[5*2, 10*2, 15*2, 20*2], dim=-1)
            
#             mean_d0, scale_d0 = torch.chunk(self.MLP_d0(torch.cat([hyper_prior, ch_ctx0, sp_ctx0], dim=-1)), chunks=2, dim=-1)
#             mean_d1, scale_d1 = torch.chunk(self.MLP_d1(torch.cat([hyper_prior, ch_ctx1, sp_ctx1], dim=-1)), chunks=2, dim=-1)
#             mean_d2, scale_d2 = torch.chunk(self.MLP_d2(torch.cat([hyper_prior, ch_ctx2, sp_ctx2], dim=-1)), chunks=2, dim=-1)
#             mean_d3, scale_d3 = torch.chunk(self.MLP_d3(torch.cat([hyper_prior, ch_ctx3, sp_ctx3], dim=-1)), chunks=2, dim=-1)
        
#             means = torch.cat([mean_d0, mean_d1, mean_d2, mean_d3], dim=-1)
#             scales = torch.cat([scale_d0, scale_d1, scale_d2, scale_d3], dim=-1)
            
#             return means, scales

#         else:
#             MLP = getattr(self, f"MLP_d{to_dec}")
#             mean, scale = torch.chunk(MLP(torch.cat([hyper_prior, ch_ctx, sp_ctx], dim=-1)), chunks=2, dim=-1)

#             return mean, scale

# class Param_Aggregation(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.MLP_d0 = nn.Sequential(
#             nn.Linear(96+5*2, 5*3),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(5*3, 5*2),
#         )

#         self.MLP_d1 = nn.Sequential(
#             nn.Linear(96+10*2, 10*3),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(10*3, 10*2),
#         )
#         self.MLP_d2 = nn.Sequential(
#             nn.Linear(96+15*2, 15*3),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(15*3, 15*2),
#         )
#         self.MLP_d3 = nn.Sequential(
#             nn.Linear(96+20*2, 20*3),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(20*3, 20*2),
#         )

#     def forward(self, hyper_prior, ch_ctx, to_dec=-1):
#         # fea_q: (nonanchor_num, knn, fea_num)
#         if to_dec == -1:
#             ch_ctx0, ch_ctx1, ch_ctx2, ch_ctx3 = torch.split(ch_ctx, split_size_or_sections=[5*2, 10*2, 15*2, 20*2], dim=-1)
            
#             mean_d0, scale_d0 = torch.chunk(self.MLP_d0(torch.cat([hyper_prior, ch_ctx0], dim=-1)), chunks=2, dim=-1)
#             mean_d1, scale_d1 = torch.chunk(self.MLP_d1(torch.cat([hyper_prior, ch_ctx1], dim=-1)), chunks=2, dim=-1)
#             mean_d2, scale_d2 = torch.chunk(self.MLP_d2(torch.cat([hyper_prior, ch_ctx2], dim=-1)), chunks=2, dim=-1)
#             mean_d3, scale_d3 = torch.chunk(self.MLP_d3(torch.cat([hyper_prior, ch_ctx3], dim=-1)), chunks=2, dim=-1)
        
#             means = torch.cat([mean_d0, mean_d1, mean_d2, mean_d3], dim=-1)
#             scales = torch.cat([scale_d0, scale_d1, scale_d2, scale_d3], dim=-1)
            
#             return means, scales

#         else:
#             MLP = getattr(self, f"MLP_d{to_dec}")
#             mean, scale = torch.chunk(MLP(torch.cat([hyper_prior, ch_ctx], dim=-1)), chunks=2, dim=-1)

#             return mean, scale

class Param_Aggregation(nn.Module):
    def __init__(self, num_groups=3, base_input=16*3*2, dims=[5, 10, 15, 20]):
        super().__init__()

        self.num_groups = num_groups
        self.num_levels = len(dims)
        self.base_input = base_input
        self.dims = dims
        self.groups = nn.ModuleList()

        for g in range(num_groups):
            level_mlps = nn.ModuleList()
            if g==0:
                dim_factor = 2
            else:
                dim_factor = 4
            for dim in dims:
                input_dim = base_input + dim * dim_factor
                hidden_dim = dim * 3
                output_dim = dim * 2
                mlp = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(hidden_dim, output_dim),
                )
                level_mlps.append(mlp)
            self.groups.append(level_mlps)


    def forward(self, hyper_prior=None, ch_ctx=None, sp_ctx=None, mask_list=[], choose_idx=None, ch_to_dec=-1, decoding=False, group_to_dec=-1):
        
        B = hyper_prior.shape[0]
        device = hyper_prior.device
        dtype = hyper_prior.dtype
    
        if ch_to_dec == -1:
            # if choose_idx is None:
            #     choose_idx = torch.ones(mask_list[0].shape[0], device=device, dtype=torch.bool)
            # 拆分 ch_ctx / sp_ctx 每层
            ch_chunks = torch.split(ch_ctx, [d * 2 for d in self.dims], dim=-1)
            sp_chunks = torch.split(sp_ctx, [d * 2 for d in self.dims], dim=-1)

            # 初始化输出 placeholder
            means = torch.zeros(B, sum([d for d in self.dims]), device=device, dtype=dtype)
            scales = torch.zeros_like(means)

            for group_id, mask in enumerate(mask_list):
                if choose_idx is not None:
                    mask = mask[choose_idx]
                if mask.sum() == 0:
                    continue  # 跳过该组无样本的情况

                h = hyper_prior[mask]
                group_means = []
                group_scales = []

                for level, dim in enumerate(self.dims):
                    ch = ch_chunks[level][mask]
                    if group_id==0:
                        x = torch.cat([h, ch], dim=-1)
                    else:
                        sp = sp_chunks[level][mask]
                        x = torch.cat([h, ch, sp], dim=-1)
                    mlp = self.groups[group_id][level]
                    mean, scale = torch.chunk(mlp(x), chunks=2, dim=-1)
                    group_means.append(mean)
                    group_scales.append(scale)

                group_means = torch.cat(group_means, dim=-1)
                group_scales = torch.cat(group_scales, dim=-1)

                means[mask] = group_means
                scales[mask] = group_scales

            return means, scales

        elif decoding and group_to_dec!=-1:
            # 单层模式
            if group_to_dec==0:
                x = torch.cat([hyper_prior, ch_ctx], dim=-1)
            else:   
                x = torch.cat([hyper_prior, ch_ctx, sp_ctx], dim=-1)
            mlp = self.groups[group_to_dec][ch_to_dec]
            means, scales = torch.chunk(mlp(x), chunks=2, dim=-1)
            return means, scales
        else:
            raise ValueError("Wrong input in Param_Aggregation")
        
class GaussianModel(nn.Module):

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self,
                 feat_dim: int=50,
                 n_offsets: int=5,
                 voxel_size: float=0.01,
                 update_depth: int=3,
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 use_feat_bank = False,
                 n_features_per_level: int=2,
                 log2_hashmap_size: int=19,
                 log2_hashmap_size_2D: int=17,
                 resolutions_list=(18, 24, 33, 44, 59, 80, 108, 148, 201, 275, 376, 514),
                 resolutions_list_2D=(130, 258, 514, 1026),
                 ste_binary: bool=True,
                 ste_multistep: bool=False,
                 add_noise: bool=False,
                 Q=1,
                 use_2D: bool=True,
                 decoded_version: bool=False,
                 is_synthetic_nerf: bool=False,
                 source_path='',
                 lmbda=0
                 ):
        super().__init__()
        print('hash_params:', use_2D, n_features_per_level,
              log2_hashmap_size, resolutions_list,
              log2_hashmap_size_2D, resolutions_list_2D,
              ste_binary, ste_multistep, add_noise)

        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank
        self.x_bound_min = torch.zeros(size=[1, 3], device='cuda')
        self.x_bound_max = torch.ones(size=[1, 3], device='cuda')
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.log2_hashmap_size_2D = log2_hashmap_size_2D
        self.resolutions_list = resolutions_list
        self.resolutions_list_2D = resolutions_list_2D
        self.ste_binary = ste_binary
        self.ste_multistep = ste_multistep
        self.add_noise = add_noise
        self.Q = Q
        self.use_2D = use_2D
        self.decoded_version = decoded_version
        self.is_synthetic_nerf = is_synthetic_nerf
        self.source_path = source_path
        self.lmbda = lmbda

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._mask = torch.empty(0)
        self._anchor_feat = torch.empty(0)

        self.opacity_accum = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)

        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)

        # self.spatial_anchor_sign = torch.empty(0)
        # self.spatial_coords = torch.empty(0)
        self.anchor_parity = "even"
        self.knn_indices = torch.empty(0)
        self.base_mask = torch.empty(0)
        self.refer_mask = torch.empty(0)
        self.morton_indices = torch.empty(0)
        self.base_0_selected = None
        self.rotation_matrix = None
        self.pca_mean = None
        self.variance = None


        # self.optimizer_feat_sp = None
        # self.optimizer_feat_hyper = None
        # self.optimizer_triplane = None
        self.optimizer = None

        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        # self.ac_sp_use2D = True
        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        mlp_input_feat_dim = feat_dim

        if self.use_2D:
            self.encoding_xyz = mix_3D2D_encoding(
                n_features=self.n_features_per_level,
                resolutions_list=self.resolutions_list,
                log2_hashmap_size=self.log2_hashmap_size,
                resolutions_list_2D=self.resolutions_list_2D,
                log2_hashmap_size_2D=self.log2_hashmap_size_2D,
                ste_binary=self.ste_binary,
                ste_multistep=self.ste_multistep,
                add_noise=self.add_noise,
                Q=self.Q,
            ).cuda()
        else:
            self.encoding_xyz = GridEncoder(
                num_dim=3,
                n_features=self.n_features_per_level,
                resolutions_list=self.resolutions_list,
                log2_hashmap_size=self.log2_hashmap_size,
                ste_binary=self.ste_binary,
                ste_multistep=self.ste_multistep,
                add_noise=self.add_noise,
                Q=self.Q,
            ).cuda()

        # self.encoding_xyz_ac = None
        encoding_params_num = 0
        for n, p in self.encoding_xyz.named_parameters():
            encoding_params_num += p.numel()
        encoding_MB = encoding_params_num / 8 / 1024 / 1024
        if not self.ste_binary: encoding_MB *= 32
        print(f'encoding_param_num={encoding_params_num}, size={encoding_MB}MB.')
        print("encoding xyz output dim:", self.encoding_xyz.output_dim)


        self.mlp_opacity = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()

        self.mlp_cov = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7*self.n_offsets),
            # nn.Linear(feat_dim, 7),
        ).cuda()

        self.mlp_color = nn.Sequential(
            nn.Linear(mlp_input_feat_dim+3+1, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()


        self.level_num = 2

        self.entropy_gaussian = Entropy_gaussian(Q=1).cuda()
        self.EG_mix_prob_2 = Entropy_gaussian_mix_prob_2(Q=1).cuda()
        self.EG_mix_prob_3 = Entropy_gaussian_mix_prob_3(Q=1).cuda()


    def get_encoding_params(self):
        params = []
        if self.use_2D:
            params.append(self.encoding_xyz.encoding_xyz.params)
            params.append(self.encoding_xyz.encoding_xy.params)
            params.append(self.encoding_xyz.encoding_xz.params)
            params.append(self.encoding_xyz.encoding_yz.params)

        else:
            params.append(self.encoding_xyz.params)

        # if self.encoding_xyz_ac is not None:
        #     if self.ac_sp_use2D:
        #         params.append(self.encoding_xyz_ac.encoding_xyz.params)
        #         params.append(self.encoding_xyz_ac.encoding_xy.params)
        #         params.append(self.encoding_xyz_ac.encoding_xz.params)
        #         params.append(self.encoding_xyz_ac.encoding_yz.params)
        #     else:
        #         params.append(self.encoding_xyz_ac.params)

        params = torch.cat(params, dim=0)
        if self.ste_binary:
            params = STE_binary.apply(params)
        return params

    def get_mlp_size(self, digit=32):
        mlp_size = 0
        for n, p in self.named_parameters():
            if 'mlp' in n:
                mlp_size += p.numel()*digit
        return mlp_size, mlp_size / 8 / 1024 / 1024

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        # self.encoding_xyz.eval()
        # self.mlp_grid.eval()
        self.mlp_deform.eval()
        self.mlp_spatial.eval() 
        self.mlp_param_aggregation.eval()
        # self.encoding_xyz_ac.eval()
        # self.mlp_ac_sp.eval()
        self.mlp_VM.eval()
        self.mlp_feat_Q.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        # self.encoding_xyz.train()
        # self.mlp_grid.train()
        self.mlp_deform.train()
        self.mlp_spatial.train()
        self.mlp_param_aggregation.train()
        # self.encoding_xyz_ac.eval()
        # self.mlp_ac_sp.train()
        self.mlp_VM.train()
        self.mlp_feat_Q.train()
        if self.use_feat_bank:
            self.mlp_feature_bank.train()

    def capture(self):
        return (
            self._anchor,
            self._offset,
            self._mask,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self._anchor,
        self._offset,
        self._mask,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        denom,
        opt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        if self.decoded_version:
            return self._scaling
        return 1.0*self.scaling_activation(self._scaling)

    @property
    def get_mask(self):
        if self.decoded_version:
            return self._mask[:, :10, :]
        mask_sig = torch.sigmoid(self._mask[:, :10, :])
        return ((mask_sig > 0.01).float() - mask_sig).detach() + mask_sig

    @property
    def get_mask_anchor(self):
        mask = self.get_mask  # [N, 10, 1]
        mask_rate = torch.mean(mask, dim=1)  # [N, 1]
        mask_anchor = ((mask_rate > 0.0).float() - mask_rate).detach() + mask_rate
        return mask_anchor  # [N, 1]

    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank

    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity

    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color

    @property
    def get_grid_mlp(self):
        return self.mlp_grid

    @property
    def get_deform_mlp(self):
        return self.mlp_deform

    @property
    def get_spatial_mlp(self):
        return self.mlp_spatial

    @property
    def get_param_mlp(self):
        return self.mlp_param_aggregation
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    # @property
    # def get_ac_sp_mlp(self):
    #     return self.mlp_ac_sp
    
    @property
    def get_VM_mlp(self):
        return self.mlp_VM

    @property
    def get_anchor(self):
        if self.decoded_version:
            return self._anchor
        anchor = torch.round(self._anchor / self.voxel_size) * self.voxel_size
        anchor = anchor.detach() + (self._anchor - self._anchor.detach())
        return anchor

    @torch.no_grad()
    def update_anchor_bound(self):
        x_bound_min = (torch.min(self._anchor, dim=0, keepdim=True)[0]).detach()
        x_bound_max = (torch.max(self._anchor, dim=0, keepdim=True)[0]).detach()
        for c in range(x_bound_min.shape[-1]):
            x_bound_min[0, c] = x_bound_min[0, c] * 1.2 if x_bound_min[0, c] < 0 else x_bound_min[0, c] * 0.8
        for c in range(x_bound_max.shape[-1]):
            x_bound_max[0, c] = x_bound_max[0, c] * 1.2 if x_bound_max[0, c] > 0 else x_bound_max[0, c] * 0.8
        self.x_bound_min = x_bound_min
        self.x_bound_max = x_bound_max
        print('anchor_bound_updated')

    def calc_interp_feat(self, x, ac_sp=False):
        # x: [N, 3]
        assert len(x.shape) == 2 and x.shape[1] == 3
        assert torch.abs(self.x_bound_min - torch.zeros(size=[1, 3], device='cuda')).mean() > 0
        x = (x - self.x_bound_min) / (self.x_bound_max - self.x_bound_min)  # to [0, 1]
        # if ac_sp:
        #     features = self.encoding_xyz_ac(x)
        # else:
        #     features = self.encoding_xyz(x)  # [N, 4*12]

        features = self.encoding_xyz(x)  # [N, 4*12]
        return features

    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        return data

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        ratio = 1
        points = pcd.points[::ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')

        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        masks = torch.ones((fused_point_cloud.shape[0], self.n_offsets+1, 1)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 6)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._mask = nn.Parameter(masks.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")


    def training_setup(self, training_args):

        self.training_args = training_args

        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        if self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._mask], 'lr': training_args.mask_lr_init * self.spatial_lr_scale, "name": "mask"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},

                # {'params': self.encoding_xyz.parameters(), 'lr': self.training_args.encoding_xyz_lr_init, "name": "encoding_xyz"},
                # {'params': self.mlp_grid.parameters(), 'lr': self.training_args.mlp_grid_lr_init, "name": "mlp_grid"},

                # {'params': self.mlp_deform.parameters(), 'lr': training_args.mlp_deform_lr_init, "name": "mlp_deform"},
                # {'params': self.mlp_param_aggregation.parameters(), 'lr': training_args.mlp_param_lr_init, "name": "mlp_param"},

                # {'params': self.mlp_VM.parameters(), 'lr': training_args.mlp_VM_lr_init, "name": "mlp_VM"},
            ]       
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._mask], 'lr': training_args.mask_lr_init * self.spatial_lr_scale, "name": "mask"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},

                # {'params': self.encoding_xyz.parameters(), 'lr': self.training_args.encoding_xyz_lr_init, "name": "encoding_xyz"},
                # {'params': self.mlp_grid.parameters(), 'lr': self.training_args.mlp_grid_lr_init, "name": "mlp_grid"},


                # {'params': self.mlp_deform.parameters(), 'lr': training_args.mlp_deform_lr_init, "name": "mlp_deform"},

                # {'params': self.mlp_param_aggregation.parameters(), 'lr': training_args.mlp_param_lr_init, "name": "mlp_param"},
                # {'params': self.mlp_VM.parameters(), 'lr': training_args.mlp_VM_lr_init, "name": "mlp_VM"},
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        self.mask_scheduler_args = get_expon_lr_func(lr_init=training_args.mask_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.mask_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.mask_lr_delay_mult,
                                                    max_steps=training_args.mask_lr_max_steps)

        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)

        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)

        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)

        self.encoding_xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.encoding_xyz_lr_init,
                                                    lr_final=training_args.encoding_xyz_lr_final,
                                                    lr_delay_mult=training_args.encoding_xyz_lr_delay_mult,
                                                    max_steps=training_args.encoding_xyz_lr_max_steps,
                                                             step_sub=0 if self.ste_binary else 10000,
                                                             )
        # self.mlp_grid_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_grid_lr_init,
        #                                             lr_final=training_args.mlp_grid_lr_final,
        #                                             lr_delay_mult=training_args.mlp_grid_lr_delay_mult,
        #                                             max_steps=training_args.mlp_grid_lr_max_steps,
        #                                                  step_sub=0 if self.ste_binary else 10000,
        #                                                  )

        self.mlp_deform_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_deform_lr_init,
                                                    lr_final=training_args.mlp_deform_lr_final,
                                                    lr_delay_mult=training_args.mlp_deform_lr_delay_mult,
                                                    max_steps=training_args.mlp_deform_lr_max_steps)

        self.mlp_param_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_param_lr_init,
                                                    lr_final=training_args.mlp_param_lr_final,
                                                    lr_delay_mult=training_args.mlp_param_lr_delay_mult,
                                                    max_steps=training_args.mlp_param_lr_max_steps)
        
        self.mlp_spatial_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_spatial_lr_init,
                                                    lr_final=training_args.mlp_spatial_lr_final,
                                                    lr_delay_mult=training_args.mlp_spatial_lr_delay_mult,
                                                    max_steps=training_args.mlp_spatial_lr_max_steps)

        self.mlp_ac_sp_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_ac_sp_lr_init,
                                                    lr_final=training_args.mlp_ac_sp_lr_final,
                                                    lr_delay_mult=training_args.mlp_ac_sp_lr_delay_mult,
                                                    max_steps=training_args.mlp_ac_sp_lr_max_steps,
                                                         step_sub=0 if self.ste_binary else 15000,
                                                         )
        # self.encoding_xyz_ac_scheduler_args = get_expon_lr_func(lr_init=training_args.encoding_xyz_ac_lr_init,
        #                                             lr_final=training_args.encoding_xyz_ac_lr_final,
        #                                             lr_delay_mult=training_args.encoding_xyz_ac_lr_delay_mult,
        #                                             max_steps=training_args.encoding_xyz_ac_lr_max_steps,
        #                                                      step_sub=0 if self.ste_binary else 15000,
        #                                                      )
        self.mlp_VM_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_VM_lr_init,
                                                    lr_final=training_args.mlp_VM_lr_final,
                                                    lr_delay_mult=training_args.mlp_VM_lr_delay_mult,
                                                    max_steps=training_args.mlp_VM_lr_max_steps,
                                                         step_sub=0 if self.ste_binary else 15000,
                                                               )
        
        self.mlp_feat_Q_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_feat_Q_lr_init,
                                                    lr_final=training_args.mlp_feat_Q_lr_final,
                                                    lr_delay_mult=training_args.mlp_feat_Q_lr_delay_mult,
                                                    max_steps=training_args.mlp_feat_Q_lr_max_steps,
                                                         step_sub=0 if self.ste_binary else 15000,
                                                               )
        
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mask":
                lr = self.mask_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_deform":
                lr = self.mlp_deform_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_param":
                lr = self.mlp_param_scheduler_args(iteration)
                param_group['lr'] = lr

            if param_group["name"] == "mlp_spatial":
                lr = self.mlp_spatial_scheduler_args(iteration)
                param_group['lr'] = lr
            # if param_group["name"] == "encoding_xyz_ac":
            #     lr = self.encoding_xyz_ac_scheduler_args(iteration)
            #     param_group['lr'] = lr
            if param_group["name"] == "mlp_ac_sp":
                lr = self.mlp_ac_sp_scheduler_args(iteration)
                param_group['lr'] = lr 

            if param_group["name"] == "mlp_VM":
                lr = self.mlp_VM_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_feat_Q":
                lr = self.mlp_feat_Q_scheduler_args(iteration)
                param_group['lr'] = lr

            # if param_group["name"] == "encoding_xyz":
            #     lr = self.encoding_xyz_scheduler_args(iteration)
            #     param_group['lr'] = lr
            # if param_group["name"] == "mlp_grid":
            #     lr = self.mlp_grid_scheduler_args(iteration)
            #     param_group['lr'] = lr

        # if iteration>10_000:
        #     for param_group in self.optimizer_triplane.param_groups:
        #         if param_group["name"] == "mlp_VM":
        #             lr = self.mlp_VM_scheduler_args(iteration)
        #             param_group['lr'] = lr

        #     for param_group in self.optimizer_feat_sp.param_groups:
        #         if param_group["name"] == "mlp_spatial":
        #             lr = self.mlp_spatial_scheduler_args(iteration)
        #             param_group['lr'] = lr
        #         if param_group["name"] == "encoding_xyz_ac":
        #             lr = self.encoding_xyz_ac_scheduler_args(iteration)
        #             param_group['lr'] = lr
        #         if param_group["name"] == "mlp_ac_sp":
        #             lr = self.mlp_ac_sp_scheduler_args(iteration)
        #             param_group['lr'] = lr 

        #     for param_group in self.optimizer_feat_hyper.param_groups:
        #         if param_group["name"] == "encoding_xyz":
        #             lr = self.encoding_xyz_scheduler_args(iteration)
        #             param_group['lr'] = lr
        #         if param_group["name"] == "mlp_grid":
        #             lr = self.mlp_grid_scheduler_args(iteration)
        #             param_group['lr'] = lr


    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._mask.shape[1]*self._mask.shape[2]):
            l.append('f_mask_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        mask = self._mask.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        N = anchor.shape[0]
        opacities = opacities[:N]
        rotation = rotation[:N]
        attributes = np.concatenate((anchor, normals, offset, mask, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))

        mask_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_mask")]
        mask_names = sorted(mask_names, key = lambda x: int(x.split('_')[-1]))
        masks = np.zeros((anchor.shape[0], len(mask_names)))
        for idx, attr_name in enumerate(mask_names):
            masks[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        masks = masks.reshape((masks.shape[0], 1, -1))

        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._mask = nn.Parameter(torch.tensor(masks, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group['name'] or 'planes' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:  # Only for opacity, rotation. But seems they two are useless?
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        temp_opacity = temp_opacity.view([-1, self.n_offsets])

        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        self.anchor_demon[anchor_visible_mask] += 1

        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter

        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)

        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or 'conv' in group['name'] or 'feat_base' in group['name'] or 'encoding' in group['name'] or 'planes' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]


        return optimizable_tensors

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._mask = optimizable_tensors["mask"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    def order_preserving_transform(self, tensor):
        # 获取排序后的值和排序后的索引
        sorted_values, sorted_indices = torch.sort(tensor)

        # 初始化一个张量来存储每个元素的排名
        rank_tensor = torch.empty_like(tensor, dtype=torch.long)

        # 使用布尔掩码避免循环
        unique_values, inverse_indices = torch.unique(sorted_values, return_inverse=True)
        rank_tensor[sorted_indices] = inverse_indices

        return rank_tensor
    
    def point_cloud_compact(self, tensor):
        compact_tensor = torch.empty_like(tensor)
        for col in range(tensor.shape[1]):
            compact_tensor[:, col] = self.order_preserving_transform(tensor[:, col])

        return compact_tensor
    
    def get_anchor_sign(self, compact_point):

        spatial_anchor_sign = (((compact_point[:, 0] % 2 == 0) & (compact_point[:, 1] % 2 == 0) & (compact_point[:, 2] % 2 == 0))|
                                ((compact_point[:, 0] % 2 == 0) & (compact_point[:, 1] % 2 == 1) & (compact_point[:, 2] % 2 == 1))|
                                ((compact_point[:, 0] % 2 == 1) & (compact_point[:, 1] % 2 == 0) & (compact_point[:, 2] % 2 == 1))|
                                ((compact_point[:, 0] % 2 == 1) & (compact_point[:, 1] % 2 == 1) & (compact_point[:, 2] % 2 == 0))
                                )
        
        return spatial_anchor_sign.cuda()

    def get_level_indices(self, points=None, valid_mask=None, reverse_indices=None, decoding=False):

        # level_0_sign = ((points[:, 0] % 2 == 1) & (points[:, 1] % 2 == 0) & (points[:, 2] % 2 == 1))

        # level_1_sign = (points[:, 0] % 2 == 0) & (points[:, 1] % 2 == 0) & (points[:, 2] % 2 == 0)

        # level_2_sign = (((points[:, 0] % 2 == 0) & (points[:, 1] % 2 == 1) & (points[:, 2] % 2 == 1))|
        #                 ((points[:, 0] % 2 == 1) & (points[:, 1] % 2 == 1) & (points[:, 2] % 2 == 0))
        #                 )

        # level_3_sign = ~(level_0_sign | level_1_sign | level_2_sign)

        # level_0_indices = level_0_sign.nonzero(as_tuple=True)[0]
        # level_1_indices = level_1_sign.nonzero(as_tuple=True)[0]
        # level_2_indices = level_2_sign.nonzero(as_tuple=True)[0]
        # level_3_indices = level_3_sign.nonzero(as_tuple=True)[0]
        
        # self.level_0_mask = torch.zeros(self.get_anchor.shape[0], dtype=torch.bool, device='cuda')
        # self.level_1_mask = torch.zeros(self.get_anchor.shape[0], dtype=torch.bool, device='cuda')
        # self.level_2_mask = torch.zeros(self.get_anchor.shape[0], dtype=torch.bool, device='cuda')
        # self.level_3_mask = torch.zeros(self.get_anchor.shape[0], dtype=torch.bool, device='cuda')

        # self.level_0_mask[valid_mask] =  level_0_sign
        # self.level_1_mask[valid_mask] =  level_1_sign
        # self.level_2_mask[valid_mask] =  level_2_sign
        # self.level_3_mask[valid_mask] =  level_3_sign

        # return [level_0_indices, level_1_indices, level_2_indices, level_3_indices]

        level_0_sign = (((points[:, 0] % 2 == 0) & (points[:, 1] % 2 == 0) & (points[:, 2] % 2 == 0))|
                        ((points[:, 0] % 2 == 0) & (points[:, 1] % 2 == 1) & (points[:, 2] % 2 == 1))|
                        ((points[:, 0] % 2 == 1) & (points[:, 1] % 2 == 0) & (points[:, 2] % 2 == 1))|
                        ((points[:, 0] % 2 == 1) & (points[:, 1] % 2 == 1) & (points[:, 2] % 2 == 0))
                        )
        level_1_sign = ~level_0_sign

        level_signs = [level_0_sign, level_1_sign]

        # level_0_sign = (points[:, 0] % 2 == 1) & (points[:, 1] % 2 == 0) & (points[:, 2] % 2 == 1)
        # level_1_sign = ((points[:, 0] % 2 == 0) & (points[:, 1] % 2 == 0) & (points[:, 2] % 2 == 0)|
        #                 (points[:, 0] % 2 == 0) & (points[:, 1] % 2 == 1) & (points[:, 2] % 2 == 1))
        # level_2_sign = ~(level_0_sign | level_1_sign)
        # level_signs = [level_0_sign, level_1_sign, level_2_sign]


        # level_0_sign = (points[:, 0] % 2 == 0) & (points[:, 1] % 2 == 0) & (points[:, 2] % 4 == 0)

        # level_1_sign = ((points[:, 0] % 2 == 0) & (points[:, 1] % 2 == 0) & (points[:, 2] % 4 == 2)|
        #                 ((points[:, 0] % 2 == 0) & (points[:, 1] % 2 == 1) & (points[:, 2] % 2 == 1))
        #                 )

        # level_2_sign = ~(level_0_sign | level_1_sign)

        # level_signs = [level_0_sign, level_1_sign, level_2_sign]
        # 初始化

        # level_0_sign = (points[:, 0] % 2 == 1) & (points[:, 1] % 2 == 0) & (points[:, 2] % 2 == 1)
        # level_1_sign = (points[:, 0] % 2 == 0) & (points[:, 1] % 2 == 0) & (points[:, 2] % 2 == 0)

        # level_2_sign = (((points[:, 0] % 2 == 0) & (points[:, 1] % 2 == 1) & (points[:, 2] % 2 == 1))|
        #                 ((points[:, 0] % 2 == 1) & (points[:, 1] % 2 == 1) & (points[:, 2] % 2 == 0))
        #                 )

        # level_3_sign = ~(level_0_sign | level_1_sign | level_2_sign)

        # level_signs = [level_0_sign, level_1_sign, level_2_sign, level_3_sign]

        # level_0_sign = (((points[:, 0] % 2 == 1) & (points[:, 1] % 2 == 0) & (points[:, 2] % 2 == 1))|
        #                 ((points[:, 0] % 2 == 0) & (points[:, 1] % 2 == 0) & (points[:, 2] % 2 == 0))
        #                 )

        # level_1_sign = (((points[:, 0] % 2 == 0) & (points[:, 1] % 2 == 1) & (points[:, 2] % 2 == 1))|
        #                 ((points[:, 0] % 2 == 1) & (points[:, 1] % 2 == 1) & (points[:, 2] % 2 == 0))
        #                 )

        # level_2_sign = ~(level_0_sign | level_1_sign)

        # level_signs = [level_0_sign, level_1_sign, level_2_sign]

        if decoding:
            level_indices = []
            for sign in level_signs:

                indices = sign.nonzero(as_tuple=True)[0]
                level_indices.append(indices)
            return level_signs, level_indices
        else:
            level_masks = []
            level_indices = []

            for sign in level_signs:
                # 记录索引
                indices = sign.nonzero(as_tuple=True)[0]
                level_indices.append(indices)

                # 创建 mask(恢复原顺序)
                mask = torch.zeros(self.get_anchor.shape[0], dtype=torch.bool, device='cuda')
                reversed_sign = sign[reverse_indices]
                mask[valid_mask] = reversed_sign
                level_masks.append(mask)

            # 将 mask 保存为 self 属性（如果你需要）
            for i, mask in enumerate(level_masks):
                setattr(self, f'level_{i}_mask', mask)

            return level_indices


    def faiss_knn(self, points, queries, knn_num=2, type='brute'):
        points = points.cpu().numpy()
        queries = queries.cpu().numpy()

        if type=='brute':
            # torch.cuda.synchronize(); t0 = time.time()
            index = faiss.IndexFlatL2(3)  # 3D 欧几里得距离
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            gpu_index.add(points)
            distances, indices = gpu_index.search(queries, knn_num)
            # torch.cuda.synchronize(); t_1 = time.time() - t0
            # print("query time:", t_1)

        elif type=='IVF':
            torch.cuda.synchronize(); t0 = time.time()
            dim, measure = 3, faiss.METRIC_L2 
            description =  'IVF4096,Flat'
            index = faiss.index_factory(dim, description, measure)
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            gpu_index.train(points)
            gpu_index.add(points)

            gpu_index.nprobe = 10
            distances, indices = gpu_index.search(queries, knn_num)
            torch.cuda.synchronize(); t_1 = time.time() - t0
            print("query time:", t_1)

        else:
            # HNSW
            torch.cuda.synchronize(); t0 = time.time()
            dim, max_nodes, measure = 3, 64, faiss.METRIC_L2   
            param =  'HNSW64' 
            # index = faiss.index_factory(dim, param, measure)
            index = faiss.IndexHNSWFlat(dim, max_nodes)
            res = faiss.StandardGpuResources()
            # gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            gpu_index = index
            gpu_index.add(points)
            gpu_index.hnsw_efSearch = 8

            distances, indices = gpu_index.search(queries, knn_num)
            torch.cuda.synchronize(); t_1 = time.time() - t0
            print("query time:", t_1)
        
        return distances, torch.from_numpy(indices).long().cuda()
    
    def voxelize(self, points):
        
        return torch.round(points / self.voxel_size)
    
    def decoding_knn(self, ancnhor, level_nums, knn=2):
        coords = self.voxelize(ancnhor)

        knn_indices_list = []
        pre_nums = 0
        for i, num in enumerate(level_nums):
            if i==0:
                pre_nums+=num
            else:
                refer_pts = coords[:pre_nums]
                query_pts = coords[pre_nums:pre_nums+num]                
                _, indices = self.faiss_knn(refer_pts, query_pts, knn)
                knn_indices_list.append(indices)

        return knn_indices_list
    

    def knn_update_3(self, knn=2):
        with torch.no_grad():
            valid_mask = self.get_mask_anchor.to(torch.bool)[:, 0]
            print("all num:", self.get_anchor.shape[0])
            coords = self.voxelize(self.get_anchor)
            valid_coords = coords[valid_mask]

            valid_num = valid_mask.sum().item()
            print("valid_num:", valid_num)
            valid_indices = torch.arange(valid_num, device='cuda').long()
            valid_indices_ori = valid_mask.nonzero(as_tuple=True)[0]
            knn_indices = torch.zeros((self.get_anchor.shape[0], knn), dtype=torch.long).cuda()
            
            ## morton sort first(encoding、decoing order):
            sorted_indices = calculate_morton_order(valid_coords)
            reverse_indices = sorted_indices.argsort()
            valid_coords = valid_coords[sorted_indices]
            valid_indices_ori = valid_indices_ori[sorted_indices]

            position_lists = self.get_level_indices(valid_coords, valid_mask, reverse_indices)
            for i, level_i in enumerate(position_lists):
                print("level num:", i, level_i.shape[0])

            decoded = None
            for i in range(len(position_lists)):
                level_i = position_lists[i]
                if i==0:
                    decoded = level_i
                else:
                    refer_pts = valid_coords[decoded]
                    query_pts = valid_coords[level_i]
                    
                    _, indices = self.faiss_knn(refer_pts, query_pts, knn)

                    refer_knn_indices = decoded[indices]
                    query_indices_ori = valid_indices_ori[level_i]
                    knn_indices[query_indices_ori] = refer_knn_indices

                    decoded = torch.cat([decoded, level_i])
                    # decoded = level_i

            self.knn_indices = knn_indices
            self.refer_mask = valid_mask
            self.morton_indices = sorted_indices

    # def morton3D_codes(self, points, bits=10, adaptive=True, safety_margin=0):
    #     with torch.no_grad():
    #         if isinstance(points, torch.Tensor):
    #             points = points.cpu().numpy().astype(np.uint64)

    #         unique_counts = [len(np.unique(points[:, i])) for i in range(3)]

    #         print("每个维度的不同值数量：", unique_counts)

    #         assert points.ndim == 2 and points.shape[1] == 3, "points should be of shape (N, 3)"
    
    #         if adaptive:
    #             max_coord = np.max(points, axis=0)
    #             max = np.max(max_coord)
    #             bits = int(np.ceil(np.log2(max) + safety_margin))
    #             print("max_coord:", max_coord)
    #             print("bits:", bits)
    #         # scale = (1 << bits) - 1
    #         # q = np.floor(points * scale).astype(np.uint32)

    #         x, y, z = points[:, 0], points[:, 1], points[:, 2]

    #         code = np.zeros_like(x, dtype=np.uint64)
    #         for i in range(bits):
    #             mask = 1 << i
    #             code |= ((x & mask) << (3*i + 2)) \
    #                 | ((y & mask) << (3*i + 1)) \
    #                 | ((z & mask) << (3*i))
    #         return code
    

    # def knn_update_2(self, patch_size=2, knn=2):
    #     with torch.no_grad():
    #         valid_mask = self.get_mask_anchor.to(torch.bool)[:, 0]
    #         print("all num:", self.get_anchor.shape[0])
    #         coords = self.voxelize(self.get_anchor)

    #         print("unique coords:", torch.unique(coords, dim=0).shape[0])
    #         # coords = coords - coords.min(dim=0).values # 平移非负
    #         valid_coords = coords[valid_mask]
    #         valid_num = valid_mask.sum().item()
    #         print("valid_num:", valid_num)
    #         valid_indices = torch.arange(valid_num, device='cuda').long()
    #         valid_indices_ori = valid_mask.nonzero(as_tuple=True)[0]
    #         knn_indices = torch.zeros((self.get_anchor.shape[0], knn), dtype=torch.long).cuda()
            
    #         # codes = self.morton3D_codes(valid_coords)
    #         # print("unique codes:", len(np.unique(codes)))
    #         # order = np.argsort(codes)
            
    #         # sorted_pts = valid_coords[order]
    #         # sorted_valid_indices_ori = valid_indices_ori[order]
    #         # sorted_valid_indices = valid_indices[order]
    #         # num_patches = valid_num // patch_size  # 忽略尾巴不足的patch
    #         # trimmed = valid_indices[:num_patches * patch_size]
    #         # left_patch = valid_indices[num_patches * patch_size:]
    #         # reshaped = trimmed.reshape((num_patches, patch_size))
    #         # position_lists = [reshaped[:, i] for i in range(patch_size)]
    #         # if left_patch.shape[0]>0:
    #         #     for i in range(left_patch.shape[0]):
    #         #         position_lists[i] = torch.cat([position_lists[i], torch.atleast_1d(left_patch[i])])

    #         valid_1 = self.get_anchor_sign(valid_coords)
    #         valid_1_indices = valid_1.nonzero(as_tuple=True)[0]
    #         valid_2 = ~valid_1
    #         valid_2_indices = valid_2.nonzero(as_tuple=True)[0]
    #         print("p_1 num:", valid_1_indices.shape[0])
    #         print("p_2 num:", valid_2_indices.shape[0])

    #         sorted_pts = valid_coords
    #         sorted_valid_indices_ori = valid_indices_ori
    #         sorted_valid_indices = valid_indices
            
    #         position_lists = [valid_1_indices, valid_2_indices]

    #         decoded = None
    #         for i in range(patch_size):
    #             patch_i = position_lists[i]
    #             if i==0:
    #                 # self.base_mask = torch.zeros(self.get_anchor.shape[0], dtype=torch.bool, device='cuda')
    #                 # base_indices_ori = sorted_valid_indices_ori[patch_i]
    #                 # self.base_mask[base_indices_ori] = True
    #                 decoded = patch_i
    #             else:
    #                 refer_pts = sorted_pts[decoded]
    #                 query_pts = sorted_pts[patch_i]
    #                 # if i==1:
    #                 #     _, indices = self.faiss_knn(refer_pts, query_pts, 1)
    #                 #     indices = indices.repeat(1, knn)
    #                 # else:
    #                 #     _, indices = self.faiss_knn(refer_pts, query_pts, knn)
                    
    #                 _, indices = self.faiss_knn(refer_pts, query_pts, knn)
    #                 refer_knn_indices = decoded[indices]
    #                 refer_knn_indices = sorted_valid_indices[refer_knn_indices]
    #                 query_indices_ori = sorted_valid_indices_ori[patch_i]
    #                 knn_indices[query_indices_ori] = refer_knn_indices

    #                 decoded = torch.cat([decoded, patch_i])

    #         # invalid_coords = coords[~valid_mask]
    #         # _, indices = self.faiss_knn(valid_coords, invalid_coords, knn)
    #         # knn_indices[~valid_mask] = indices

    #         # base_coords = coords[self.base_mask]
    #         # non_base_valid_mask = ~(self.base_mask[valid_mask])
    #         # non_base_valid_coords = valid_coords[valid_1]
    #         # non_base_valid_indices = valid_indices[non_base_valid_mask]

    #         # _, indices = self.faiss_knn(non_base_valid_coords, base_coords, knn)
    #         # knn_indices[self.base_mask] = non_base_valid_indices[indices]

    #         refer_pts = valid_coords[valid_2]
    #         query_pts = valid_coords[valid_1]

    #         _, indices = self.faiss_knn(refer_pts, query_pts, knn)
    #         refer_knn_indices = valid_2_indices[indices]
    #         query_indices_ori = valid_indices_ori[valid_1]
    #         knn_indices[query_indices_ori] = refer_knn_indices   

    #         refer_pts = valid_coords[valid_1]
    #         query_pts = coords[~valid_mask]
    #         _, indices = self.faiss_knn(refer_pts, query_pts, knn)
    #         refer_knn_indices = valid_1_indices[indices]
    #         knn_indices[~valid_mask] = refer_knn_indices


    #         self.knn_indices = knn_indices

    #         self.refer_mask = valid_mask

    # def knn_update_1(self, base_ratio=4, knn=2):
    #     with torch.no_grad():
    #         valid_mask = self.get_mask_anchor.to(torch.bool)[:, 0]
    #         coords = self.voxelize(self.get_anchor)
    #         valid_coords = coords[valid_mask]
    #         valid_num = valid_mask.sum().item()
    #         print("valid_num:", valid_num)
    #         valid_indices = torch.arange(valid_num, device='cuda').long()
    #         valid_indices_ori = valid_mask.nonzero(as_tuple=True)[0]
    #         knn_indices = torch.zeros((self.get_anchor.shape[0], knn), dtype=torch.long).cuda()
            
    #         levels = [valid_num//base_ratio, valid_num//base_ratio]
    #         while(True):
    #             refer_num = sum(levels)
    #             left_num = valid_num - refer_num
    #             # query_num = refer_num//2
    #             # if (left_num-query_num)<= levels[0]:
    #             query_num = levels[0]//2
    #             if left_num<= query_num:
    #                 levels.append(left_num)
    #                 break
    #             else:
    #                 levels.append(query_num)
    #         print("levels:", levels)
    #         # 0 : 1/16 + 1/16
    #         # 1: 2/16
    #         # 2: 2/16
    #         # 3: 3/16
    #         # 4: 4.5/16
    #         # 5：rest（2.5/16）

    #         # level_0_1_2 = valid_num//8
    #         # level_3 = valid_num*3//16
    #         # level_4 = valid_num*9//32
    #         # level_5 = valid_num - level_0_1_2*3 - level_3 -level_4
    #         # levels = [level_0_1_2, level_0_1_2, level_0_1_2, level_3, level_4, level_5]

    #         # 0 : 4/16
    #         # 1: 2/16
    #         # 2: 3/16
    #         # 3: 4.5/16
    #         # 4：rest（2.5/16）
    #         # level_0 = valid_num//4
    #         # level_1 = valid_num//8
    #         # level_2 = valid_num*3//16
    #         # level_3 = valid_num*9//32
    #         # level_4 = valid_num - level_3 -level_2 - level_1 - level_0
    #         # levels = [level_0, level_1, level_2, level_3, level_4] 

    #         refer_mask = torch.zeros(valid_num, dtype=torch.bool, device='cuda')
    #         # base_num_0 = levels[0]//2
    #         base_num_0 = levels[0]
    #         if self.base_0_selected is None:
    #             base_0_selected = torch.randperm(valid_num)[:base_num_0]
    #             self.base_0_selected = torch.zeros(self.get_anchor.shape[0], dtype=torch.bool, device='cuda')
    #             final_indices= valid_indices_ori[base_0_selected]
    #             self.base_0_selected[final_indices] = True
    #             print("num_1:", self.base_0_selected[valid_mask].sum().item())
    #         else:
    #             if self.base_0_selected[valid_mask].sum().item() < base_num_0:
    #                 print("less than!")

    #                 append_num = base_num_0 - self.base_0_selected[valid_mask].sum().item()
    #                 not_selected_num = valid_num - self.base_0_selected[valid_mask].sum().item()
    #                 not_selected_indices = torch.where(~self.base_0_selected[valid_mask])[0]
    #                 append_indices = not_selected_indices[torch.randperm(not_selected_num)[:append_num]]
    #                 final_indices= valid_indices_ori[append_indices]
    #                 self.base_0_selected[final_indices] = True

    #             print("base_valid_num:", self.base_0_selected[valid_mask].sum().item())
    #             base_index = self.base_0_selected[valid_mask].nonzero(as_tuple=True)[0]
    #             print("base_index shape:", base_index.shape)
    #             base_0_selected = base_index[:base_num_0]
    #             print("selected base num:", base_0_selected.shape)
    #         refer_mask[base_0_selected] = True
    #         base_indices_ori = valid_indices_ori[refer_mask]
    #         self.base_mask = torch.zeros(self.get_anchor.shape[0], dtype=torch.bool, device='cuda')
    #         self.base_mask[base_indices_ori] = True

    #         # for i, level_num in enumerate(levels):
    #             # if i==0:
    #             #     refer_coords = valid_coords[refer_mask]
    #             #     query_coords = valid_coords[~refer_mask]
    #             #     query_indices = valid_indices[~refer_mask]
    #             #     distances, _ = self.faiss_knn(refer_coords, query_coords, 1)
    #             #     base_num_1 = levels[0]-base_num_0
    #             #     nearest_k_indices = np.argsort(distances[:, 0])[:base_num_1]
    #             #     k_query_indices = query_indices[nearest_k_indices]
    #             #     refer_mask[k_query_indices] = True
    #             #     base_indices_ori = valid_indices_ori[refer_mask]
    #             #     self.base_mask = torch.zeros(self.get_anchor.shape[0], dtype=torch.bool, device='cuda')
    #             #     self.base_mask[base_indices_ori] = True
    #             # else:
    #             #     refer_coords = valid_coords[refer_mask]
    #             #     refer_indices = valid_indices[refer_mask]
    #             #     query_coords = valid_coords[~refer_mask]
    #             #     query_indices = valid_indices[~refer_mask]
    #             #     distances, indices = self.faiss_knn(refer_coords, query_coords, knn)
    #             #     # avg_distances = np.mean(distances, axis=1)
    #             #     # nearest_k_indices = np.argsort(avg_distances)[:level_num]
    #             #     nearest_k_indices = np.argsort(distances[:, 0])[:level_num]
    #             #     k_query_indices = query_indices[nearest_k_indices]
    #             #     k_query_indices_ori = valid_indices_ori[k_query_indices]
    #             #     tmp_indices = indices[nearest_k_indices]
    #             #     k_refer_indices = refer_indices[tmp_indices]
    #             #     knn_indices[k_query_indices_ori] = k_refer_indices
    #             #     refer_mask[k_query_indices] = True

    #         for i, level_num in enumerate(levels[1:]):
    #             refer_coords = valid_coords[refer_mask]
    #             refer_indices = valid_indices[refer_mask]
    #             query_coords = valid_coords[~refer_mask]
    #             query_indices = valid_indices[~refer_mask]
    #             if i==0:
    #                 distances, indices = self.faiss_knn(refer_coords, query_coords, 1)
    #                 indices = indices.repeat(1, knn)
    #             else:
    #                 distances, indices = self.faiss_knn(refer_coords, query_coords, knn)
    #             # avg_distances = np.mean(distances, axis=1)
    #             # nearest_k_indices = np.argsort(avg_distances)[:level_num]
    #             nearest_k_indices = np.argsort(distances[:, 0])[:level_num]
    #             k_query_indices = query_indices[nearest_k_indices]
    #             k_query_indices_ori = valid_indices_ori[k_query_indices]
    #             tmp_indices = indices[nearest_k_indices]
    #             k_refer_indices = refer_indices[tmp_indices]
    #             knn_indices[k_query_indices_ori] = k_refer_indices
    #             refer_mask[k_query_indices] = True
    #             # print("iteration:", i)
    #             # print("refer_num:", refer_mask.sum().item())

    #         invalid_coords = coords[~valid_mask]
    #         _, indices = self.faiss_knn(valid_coords, invalid_coords, knn)
    #         knn_indices[~valid_mask] = indices

    #         self.knn_indices = knn_indices

    #         self.refer_mask = valid_mask

    def knn_update_0(self, knn=2):
        with torch.no_grad():
            # prune
            valid_mask = self.get_mask_anchor.to(torch.bool)[:, 0]
            # valid_anchor = self.get_anchor[valid_mask]
            # optimizable_tensors = self._prune_anchor_optimizer(valid_mask)

            # self._anchor = optimizable_tensors["anchor"]
            # self._offset = optimizable_tensors["offset"]
            # self._mask = optimizable_tensors["mask"]
            # self._anchor_feat = optimizable_tensors["anchor_feat"]
            # self._opacity = optimizable_tensors["opacity"]
            # self._scaling = optimizable_tensors["scaling"]
            # self._rotation = optimizable_tensors["rotation"]
            
            # self.spatial_coords = self.point_cloud_compact(valid_anchor)
            self.spatial_coords = self.voxelize(self.get_anchor)
        
            # update 
            if self.anchor_parity == "even":
                self.spatial_anchor_sign = torch.logical_and(self.get_anchor_sign(self.spatial_coords), valid_mask)
            else:
                self.spatial_anchor_sign = torch.logical_and(~(self.get_anchor_sign(self.spatial_coords)), valid_mask)
            
            anchor_coords = self.spatial_coords[self.spatial_anchor_sign]
            nonanchor_coords = self.spatial_coords[~self.spatial_anchor_sign]
            print("all num", self.get_anchor.shape[0])
            print("anchor num", anchor_coords.shape[0])
            print("invalid num", (~valid_mask).sum().item())
            print("valid nonanchor num", nonanchor_coords.shape[0]-(~valid_mask).sum().item())


            # _, indices = self.faiss_knn(anchor_coords, nonanchor_coords)
            # original_ac_indices = self.spatial_anchor_sign.nonzero(as_tuple=True)[0]
            # indices = original_ac_indices[indices]
            # self.knn_indices = indices.long()
            
            # valid_coords = self.spatial_coords[valid_mask]
            # _, indices = self.faiss_knn(valid_coords, self.spatial_coords)
            # original_valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            # indices = original_valid_indices[indices]
            # self.knn_indices = indices.long()[..., 1:]

            valid_nac_sign = torch.logical_and(~self.spatial_anchor_sign, valid_mask)
            valid_nac_coords = self.spatial_coords[valid_nac_sign]
            _, ac_indices = self.faiss_knn(valid_nac_coords, anchor_coords, knn_num=knn)
            _, nac_indices = self.faiss_knn(anchor_coords, nonanchor_coords, knn_num=knn)

            valid_nac_indices_in_original = valid_nac_sign.nonzero(as_tuple=True)[0]
            ac_indices_in_original = self.spatial_anchor_sign.nonzero(as_tuple=True)[0]
            ac_indices = valid_nac_indices_in_original[ac_indices]
            nac_indices = ac_indices_in_original[nac_indices]
            # 注意knn_indices的形状
            knn_indices = torch.empty((self.spatial_coords.shape[0], knn), dtype=torch.long).cuda()
            knn_indices[self.spatial_anchor_sign] = ac_indices
            knn_indices[~self.spatial_anchor_sign] = nac_indices

            self.knn_indices = knn_indices


            # statis
            # self.opacity_accum = self.opacity_accum[valid_mask]
            # self.anchor_demon = self.anchor_demon[valid_mask]

            # offset_gradient_accum = self.offset_gradient_accum.view(-1, self.n_offsets)
            # self.offset_gradient_accum = offset_gradient_accum[valid_mask].view(-1, 1)
            # offset_denom = self.offset_denom.view(-1, self.n_offsets)
            # self.offset_denom = offset_denom[valid_mask].view(-1, 1)

    def spatial_efficiency(self, anchor_coords, nonanchor_coords):

        anchor_num = anchor_coords.shape[0]
        nonanchor_num = nonanchor_coords.shape[0]
        all_num = anchor_num+nonanchor_num
        all_coords = torch.cat([anchor_coords, nonanchor_coords], dim=0).cuda()
        batch = torch.zeros(all_num, 1).cuda().int()
        all_coords = torch.cat([batch, all_coords], dim=1)
        all_fea = torch.ones((all_coords.shape[0], 1), dtype=torch.float32)
        all_fea[anchor_num:, :] = 0

        kernel_size = 25
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        all_sparse = ME.SparseTensor(features=all_fea, coordinates=all_coords, device=device)
        conv = ME.MinkowskiConvolution(1, 1, kernel_size=kernel_size, stride=1, bias=False, dimension=3).cuda()
        conv.kernel.data.fill_(1)
        output = conv(all_sparse)


        anchor_per_nonanchor =  output.F[anchor_num:, :]
        average_per_kernel = torch.sum(anchor_per_nonanchor)/nonanchor_num

        return anchor_per_nonanchor, average_per_kernel
    def apply_pca_transformation(self):
        """
        Applies PCA transformation to the input data x using PyTorch.
        """
        # if self.mean_x is not None:
        #     mean_x = self.mean_x
        #     std_x = self.std_x
        #     R_x = self.R_x
        #     normalized_x = (x - mean_x) / std_x
        # else:
        mask_anchor = self.get_mask_anchor.to(torch.bool)[:, 0]  # N
        anchornp = self.get_anchor[mask_anchor].detach().cpu().numpy()

        x = anchornp[:,0]
        y = anchornp[:,1]
        z = anchornp[:,2]
        xyz = np.vstack([x, y, z]).T
        lof = LocalOutlierFactor(n_neighbors=50, contamination=0.05)
        labels = lof.fit_predict(xyz)

        dense_points = torch.tensor(xyz[labels == 1], dtype=torch.float32).cuda()

        mean = torch.mean(dense_points, dim=0)
        centered_points = dense_points - mean
        cov_matrix  = torch.cov(centered_points.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        self.rotation_matrix = nn.Parameter(sorted_eigenvectors, requires_grad=False)
        self.pca_mean = nn.Parameter(mean, requires_grad=False)
        self.variance = nn.Parameter(torch.sqrt(eigenvalues[sorted_indices]), requires_grad=False)


        # mean_x = torch.mean(x, dim=0)
        # std_x = torch.std(x, dim=0)
        # normalized_x = (x - mean_x) / std_x  # Normalize
        
        # cov_matrix = torch.mm(normalized_x.T, normalized_x) / x.shape[0]
        # eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)  # Compute eigenvalues and eigenvectors
        
        # sorted_indices = torch.argsort(eigenvalues, descending=True)
        # R_x = eigenvectors[:, sorted_indices]  # Rotation matrix


    def contract(self, pts):
        """
        Apply the contraction function to map x into a bounded space (radius <= 2).
        """
        pts = pts - self.pca_mean
        pts = torch.matmul(pts, self.rotation_matrix.detach())
        pts = pts / self.variance.detach()

        ord = 2
        mag = torch.linalg.norm(pts, ord=ord, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1
        mask = mask.unsqueeze(-1) + 0.0
        x_c = (2 - 1 / mag) * (pts / mag)
        x = x_c * mask + pts * (1 - mask)
        # x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
        x = x / 2 # [-inf, inf] is at [-1, 1]
        return x
        # norm_x = torch.norm(pts, dim=1, keepdim=True)
        # factor = torch.where(norm_x <= 1, torch.ones_like(norm_x), (2 - 1 / norm_x))
        # return factor * (pts / norm_x)*0.5 # [-1, 1]

    def process_data(self, x, update_R=False):
        """
        Full pipeline: PCA transformation and contraction function using PyTorch.
        """
        if self.rotation_matrix is None or update_R:
            self.apply_pca_transformation()

        contracted_x = self.contract(x)
        return contracted_x

    def query_triplane(self, x, update_R=False, quant_step=2**4, training=True):
        """
        Queries a triplane representation given 3D points after PCA and contraction.

        triplane_features: (3, C, H, W)  # 3 个特征平面，每个是 C x H x W
        x: (N, 3)  # 查询的 3D 坐标
        """
        output = []
        # **Step 1: 先进行 PCA 变换 + contraction 收缩**
        x = self.process_data(x, update_R)  # 先进行 PCA 对齐，再收缩，结果在 [-2, 2] 之间

        # **Step 2: 坐标归一化到 [0, H] 和 [0, W]**
        indices2D = torch.stack(
                (x[..., [0, 1]], x[..., [0, 2]], x[..., [1, 2]]),
                dim=0,
            )  # (3, N, 2)

        # **Step 3: 进行双线性插值**
        def bilinear_sample(triplanes, indices):
            indices = indices.view(3, -1, 1, 2)  # (3, N, 1, 2)
            return F.grid_sample(
                triplanes,
                indices, 
                mode='bilinear', 
                align_corners=True, padding_mode='border')
        
        for triplane in self.triplanes:
            N, C, H, W = triplane.shape
            if training:
                quantized_plane = quantize_ste(triplane*quant_step)/quant_step
            else:
                quantized_plane = torch.round(triplane*quant_step)/quant_step
            out = bilinear_sample(quantized_plane, indices2D)
            out = rearrange(out, "Np Cp N () -> N (Np Cp)", Np=3) # cat 3 plane features
            output.append(out)
        
        # if len(output)==1:
        #     return self.mlp_VM(output[0])
        # else:
        #     return self.mlp_VM(torch.cat(output, dim=-1))

        if len(output)==1:
            return output[0]
        else:
            return torch.cat(output, dim=-1)

    def query_VM(self, x, L=10, update_R=False):
        output = []
        x = self.process_data(x, update_R)
        indices2D = x[..., [0, 1]] # (N, 2)

        # **Step 3: 进行双线性插值**
        def bilinear_sample(plane, indices):
            indices = indices.view(1, -1, 1, 2)  # (1, N, 1, 2)
            plane = plane.unsqueeze(0)
            return F.grid_sample(
                plane,
                indices, 
                mode='bilinear', 
                align_corners=True, padding_mode='border')
        
        for plane in self.planes:
            C, H, W = plane.shape

            out = bilinear_sample(plane, indices2D)
            out = out.squeeze().T
            output.append(out)

        z = x[..., 2:3]
        frequencies = 2 ** torch.arange(L, dtype=torch.float32).cuda() * math.pi
        interleaved  = torch.stack([torch.sin(frequencies * z), torch.cos(frequencies * z)], dim=-1).flatten(start_dim=-2)  #(N, 2L)

        return self.mlp_VM(torch.cat(output + [interleaved], dim=-1))
        
    def sp_config(self):

        # if self.ac_sp_use2D:
        #     self.encoding_xyz_ac = mix_3D2D_encoding(
        #         n_features=self.n_features_per_level,
        #         resolutions_list=self.resolutions_list,
        #         log2_hashmap_size=self.log2_hashmap_size,
        #         resolutions_list_2D=self.resolutions_list_2D,
        #         log2_hashmap_size_2D=self.log2_hashmap_size_2D,
        #         ste_binary=self.ste_binary,
        #         ste_multistep=self.ste_multistep,
        #         add_noise=self.add_noise,
        #         Q=self.Q,
        #     ).cuda()
        # else:
        #     self.encoding_xyz_ac = GridEncoder(
        #         num_dim=3,
        #         n_features=self.n_features_per_level,
        #         resolutions_list=self.resolutions_list,
        #         log2_hashmap_size=self.log2_hashmap_size,
        #         ste_binary=self.ste_binary,
        #         ste_multistep=self.ste_multistep,
        #         add_noise=self.add_noise,
        #         Q=self.Q,
        #     ).cuda()
        # self.mlp_ac_sp = nn.Linear(self.encoding_xyz_ac.output_dim, 50*2).cuda()
        # self.mlp_ac_sp = nn.Sequential(
        #     nn.Linear(self.encoding_xyz_ac.output_dim, 50*2),
        #     nn.ReLU(True),
        #     nn.Linear(50*2, 50*2),
        # ).cuda()

        # self.mlp_base_sp = nn.Sequential(
        #     nn.Linear(3,  50*2),
        #     nn.ReLU(True),
        #     nn.Linear(50*2, 50*2),
        # ).cuda()

        self.mlp_spatial = Spatial_CTX(num_groups=self.level_num, knn=2).cuda()
        # self.mlp_spatial = Spatial_CTX().cuda()
        # self.mlp_spatial =  Spatial_CTX_fea_knn_tiny().cuda()
        nac_sp_params = {'params': self.mlp_spatial.parameters(), 'lr': self.training_args.mlp_spatial_lr_init, "name": "mlp_spatial"}
        # base_sp_params =  {'params': self.mlp_base_sp.parameters(), 'lr': self.training_args.mlp_ac_sp_lr_init, "name": "mlp_ac_sp"}
        # encoding_xyz_ac_params = {'params': self.encoding_xyz_ac.parameters(), 'lr': self.training_args.encoding_xyz_ac_lr_init, "name": "encoding_xyz_ac"}
        self.optimizer.add_param_group(nac_sp_params)
        # self.optimizer.add_param_group(base_sp_params)
        
        # feat_sp_l = [{'params': self.mlp_spatial.parameters(), 'lr': self.training_args.mlp_spatial_lr_init, "name": "mlp_spatial"}]
        # self.optimizer_feat_sp = torch.optim.Adam(feat_sp_l, lr=0.0, eps=1e-15)
    def ch_config(self):
        if not self.is_synthetic_nerf:
            # self.mlp_deform = Channel_CTX().cuda()
            self.mlp_deform = Channel_CTX(num_groups=self.level_num).cuda()
        else:
            print('find synthetic nerf, use Channel_CTX_fea_tiny')
            self.mlp_deform = Channel_CTX_fea_tiny().cuda()
        ch_params = {'params': self.mlp_deform.parameters(), 'lr': self.training_args.mlp_deform_lr_init, "name": "mlp_deform"}
        self.optimizer.add_param_group(ch_params)

    def param_config(self):
        self.mlp_param_aggregation = Param_Aggregation(num_groups=self.level_num).cuda()
        param_params = {'params': self.mlp_param_aggregation.parameters(), 'lr': self.training_args.mlp_param_lr_init, "name": "mlp_param"}
        self.optimizer.add_param_group(param_params)

    def encoding_xyz_config(self):

        # self.mlp_grid = nn.Sequential(
        #     nn.Linear(self.encoding_xyz.output_dim, self.feat_dim*2),
        #     nn.ReLU(True),
        #     nn.Linear(self.feat_dim*2, (self.feat_dim+6+3*self.n_offsets)*2+1+1+1),
        # ).cuda()

        self.mlp_grid = nn.Linear(self.encoding_xyz.output_dim, self.feat_dim*2).cuda()

        encoding_xyz_params = {'params': self.encoding_xyz.parameters(), 'lr': self.training_args.encoding_xyz_lr_init, "name": "encoding_xyz"}
        mlp_grid_params = {'params': self.mlp_grid.parameters(), 'lr': self.training_args.mlp_grid_lr_init, "name": "mlp_grid"}
        self.optimizer.add_param_group(encoding_xyz_params)
        self.optimizer.add_param_group(mlp_grid_params)

        # feat_hyper_l = [
        #     {'params': self.encoding_xyz.parameters(), 'lr': self.training_args.encoding_xyz_lr_init, "name": "encoding_xyz"},
        #     {'params': self.mlp_grid.parameters(), 'lr': self.training_args.mlp_grid_lr_init, "name": "mlp_grid"}
        # ]
        # self.optimizer_feat_hyper = torch.optim.Adam(feat_hyper_l, lr=0.0, eps=1e-15)
    def save_triplanes(self, quant_step=2**4):
        # A: [3, C, R, R]
        # B: [3, C, 2R, 2R]
        A = torch.round(self.triplanes[0]*quant_step)
        N, C, R, R = A.shape
        B = torch.round(self.triplanes[1]*quant_step)
        # 拆分 B 为 4 块大小为 [C, R, R]
        B_split = []

        # 按空间 2x2 划分
        for i in range(2):  # 行
            for j in range(2):  # 列
                patch = B[:, :, i*R:(i+1)*R, j*R:(j+1)*R]  # [3, C, R, R]
                B_split.append(patch)

        # B_split: List of 4 tensors of shape [3, C, R, R]
        B_split = torch.stack(B_split, dim=1)  # [3, 4, C, R, R]
        B_split = B_split.view(-1, C, R, R)  # [12, C, R, R]

        # 合并 A 和 B_split
        merged = torch.cat([A, B_split], dim=0)  # [15, C, R, R]

        # 保存为 .pt
        dataset = self.source_path.split('/')[-2]
        scene = self.source_path.split('/')[-1]
        lmbda = self.lmbda
        filename = f'./triplanes/{dataset}_{scene}_{lmbda}.pt'
        torch.save(merged, filename)

    def get_triplanes(self, quant_step=2**4):
        # 加载保存的合并张量
        dataset = self.source_path.split('/')[-2]
        scene = self.source_path.split('/')[-1]
        lmbda = self.lmbda
        filename = f'./triplanes/{dataset}_{scene}_{lmbda}.pt'
        merged = torch.load(filename).cuda()  # [15, C, R, R]

        # 还原 A
        A_recovered = merged[:3]/quant_step  # [3, C, R, R]

        # 还原 B：将 12 个 patch 拼回 3 个 plane，每个 plane 是 [C, 2R, 2R]
        C = merged.shape[1]
        R = merged.shape[2]

        B_patches = merged[3:]  # [12, C, R, R]
        B_planes = []

        for i in range(3):  # 对每个 plane 做拼接
            # 取出这个 plane 的四块
            tl = B_patches[i*4 + 0]  # top-left
            tr = B_patches[i*4 + 1]  # top-right
            bl = B_patches[i*4 + 2]  # bottom-left
            br = B_patches[i*4 + 3]  # bottom-right

            # 水平拼 top、bottom
            top = torch.cat([tl, tr], dim=2)      # [C, R, 2R]
            bottom = torch.cat([bl, br], dim=2)   # [C, R, 2R]

            # 垂直拼接 top 和 bottom
            full = torch.cat([top, bottom], dim=1)  # [C, 2R, 2R]

            B_planes.append(full)

        # 堆叠还原 B
        B_recovered = torch.stack(B_planes, dim=0)/quant_step  # [3, C, 2R, 2R]

        self.triplanes = [A_recovered, B_recovered]

    def triplane_config(self, base_res=64, triplane_num=2, triplane_ch=16, triplane_lr=0.1, a=-0.25, b=0.25):
        triplanes = nn.ParameterList()
        anchor_num=self.get_anchor.shape[0]
        base_res=round((anchor_num / 36) ** 0.5)
        base_res = max(64, min(base_res, 128))
        base_res = math.ceil(base_res / 4) * 4 # for entropy conv
        for i in range(triplane_num):
            res = base_res*(i+1)
            triplane = nn.Parameter(torch.empty([3, triplane_ch, res, res], dtype=torch.float32, device="cuda")) 
            nn.init.uniform_(triplane, a=a, b=b) 
            triplanes.append(triplane)
        
        self.triplanes = triplanes

        self.mlp_VM = nn.Sequential(
            nn.Linear(16*3*2, self.feat_dim*2),
            nn.ReLU(True),
            nn.Linear(self.feat_dim*2, (6+3*self.n_offsets)*2+1+1),
        ).cuda()

        self.mlp_feat_Q = nn.Linear(16*3*2, 1).cuda()

        triplanes_params = {'params': self.triplanes , 'lr': triplane_lr, "name": "triplanes"}
        VM_params = {'params': self.mlp_VM.parameters(), 'lr': self.training_args.mlp_VM_lr_init, "name": "mlp_VM"}
        feat_Q_params =  {'params': self.mlp_feat_Q.parameters(), 'lr': self.training_args.mlp_feat_Q_lr_init, "name": "mlp_feat_Q"}
        self.optimizer.add_param_group(triplanes_params)
        self.optimizer.add_param_group(VM_params)
        self.optimizer.add_param_group(feat_Q_params)
        
        # triplane_l = [
        #     {'params': self.triplanes , 'lr': triplane_lr, "name": "triplanes"},
        #     {'params': self.mlp_VM.parameters(), 'lr': self.training_args.mlp_VM_lr_init, "name": "mlp_VM"}
        # ]
        # self.optimizer_triplane = torch.optim.Adam(triplane_l, lr=0.0, eps=1e-15)

    def VM_config(self, base_res=64, plane_num=2, plane_ch=16, plane_lr=0.0075, a=-0.01, b=0.01):
        planes = nn.ParameterList()
        anchor_num=self.get_anchor.shape[0]
        base_res=round((anchor_num / 36) ** 0.5)
        base_res = max(64, min(base_res, 128))

        for i in range(plane_num):
            res = base_res*(i+1)
            plane = nn.Parameter(torch.empty([plane_ch, res, res], dtype=torch.float32, device="cuda"))
            nn.init.uniform_(plane, a=a, b=b)
            planes.append(plane)
        self.planes = planes
        
        
        self.mlp_VM = nn.Sequential(
            nn.Linear(16*2+20, self.feat_dim*2),
            nn.ReLU(True),
            nn.Linear(self.feat_dim*2, (self.feat_dim+6+3*self.n_offsets)*2+1+1+1),
        ).cuda()
        # triplanes_params = {'params': self.planes , 'lr': plane_lr, "name": "planes"}
        # VM_params = {'params': self.mlp_VM.parameters(), 'lr': self.training_args.mlp_VM_lr_init, "name": "mlp_VM"}
        # self.optimizer.add_param_group(triplanes_params)
        # self.optimizer.add_param_group(VM_params)

    def anchor_growing(self, grads, threshold, offset_mask):
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):  # 3
            # for self.update_depth=3, self.update_hierachy_factor=4: 2**0, 2**1, 2**2
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)

            rand_mask = torch.rand_like(candidate_mask.float()) > (0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)

            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)
            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:, :3].unsqueeze(dim=1)

            # for self.update_depth=3, self.update_hierachy_factor=4: 4**0, 4**1, 4**2
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            size_factor = size_factor+1 if size_factor%2==0 else size_factor

            cur_size = self.voxel_size*size_factor

            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)
            use_chunk = True
            if selected_grid_coords_unique.numel() == 0:
                candidate_anchor = torch.empty((0, grid_coords.shape[1]), device=grid_coords.device)
            else:
                if use_chunk:
                    chunk_size = 4096
                    max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                    remove_duplicates_list = []
                    for i in range(max_iters):
                        # print(f"selected shape: {selected_grid_coords_unique.shape}, i={i}")
                        cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                        remove_duplicates_list.append(cur_remove_duplicates)

                    remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
                else:
                    remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

                remove_duplicates = ~remove_duplicates
                candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1, 2]).float().cuda() * cur_size
                new_scaling = torch.log(new_scaling)

                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:, 0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).float().cuda()
                new_masks = torch.ones_like(candidate_anchor[:, 0:1]).unsqueeze(dim=1).repeat([1, self.n_offsets+1, 1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "mask": new_masks,
                    "opacity": new_opacities,
                }

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()

                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._mask = optimizable_tensors["mask"]
                self._opacity = optimizable_tensors["opacity"]

    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)

        self.anchor_growing(grads_norm, grad_threshold, offset_mask)

        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)

        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask)  # [N]

        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum

        # update opacity accum
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()

        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)

        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def save_mlp_checkpoints(self,path):
        mkdir_p(os.path.dirname(path))

        if self.use_feat_bank:
            torch.save({
                'opacity_mlp': self.mlp_opacity.state_dict(),
                'mlp_feature_bank': self.mlp_feature_bank.state_dict(),
                'cov_mlp': self.mlp_cov.state_dict(),
                'color_mlp': self.mlp_color.state_dict(),
                # 'encoding_xyz': self.encoding_xyz.state_dict(),
                # 'grid_mlp': self.mlp_grid.state_dict(),
                # 'deform_mlp': self.mlp_deform.state_dict(),
                # 'spatial_mlp': self.mlp_spatial.state_dict(),
            }, path)
        else:
            torch.save({
                'opacity_mlp': self.mlp_opacity.state_dict(),
                'cov_mlp': self.mlp_cov.state_dict(),
                'color_mlp': self.mlp_color.state_dict(),
                # 'encoding_xyz': self.encoding_xyz.state_dict(),
                # 'grid_mlp': self.mlp_grid.state_dict(),
                # 'deform_mlp': self.mlp_deform.state_dict(),
                # 'spatial_mlp': self.mlp_spatial.state_dict(),
            }, path)


    def load_mlp_checkpoints(self,path):
        checkpoint = torch.load(path)
        self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
        self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
        self.mlp_color.load_state_dict(checkpoint['color_mlp'])
        if self.use_feat_bank:
            self.mlp_feature_bank.load_state_dict(checkpoint['mlp_feature_bank'])
        # self.encoding_xyz.load_state_dict(checkpoint['encoding_xyz'])
        # self.mlp_grid.load_state_dict(checkpoint['grid_mlp'])
        # self.mlp_deform.load_state_dict(checkpoint['deform_mlp'])
        # self.mlp_spatial.load_state_dict(checkpoint['spatial_mlp'])

    def contract_to_unisphere(self,
        x: torch.Tensor,
        aabb: torch.Tensor,
        ord: int = 2,
        eps: float = 1e-6,
        derivative: bool = False,
    ):
        aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1

        if derivative:
            dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
                1 / mag**3 - (2 * mag - 1) / mag**4
            )
            dev[~mask] = 1.0
            dev = torch.clamp(dev, min=eps)
            return dev
        else:
            mask = mask.unsqueeze(-1) + 0.0
            x_c = (2 - 1 / mag) * (x / mag)
            x = x_c * mask + x * (1 - mask)
            x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
            return x

    @torch.no_grad()
    def estimate_final_bits(self):

        # Q_feat = 1
        # Q_scaling = 0.001
        # Q_offsets = 0.2

        # mask_anchor = self.get_mask_anchor.to(torch.bool)[:, 0]  # N

        # _anchor = self.get_anchor[mask_anchor]
        # _feat = self._anchor_feat[mask_anchor]
        # _grid_offsets = self._offset[mask_anchor]
        # _scaling = self.get_scaling[mask_anchor]
        # _mask = self.get_mask[mask_anchor]
        # hash_embeddings = self.get_encoding_params()

        # feat_context = self.calc_interp_feat(_anchor)  # [N_visible_anchor*0.2, 32]
        # mean, scale, prob, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
        #     torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[self.feat_dim, self.feat_dim, self.feat_dim, 6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1, 1], dim=-1)  # [N_visible_anchor, 32], [N_visible_anchor, 32]
        # Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
        # Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
        # Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))
        # _feat = (STE_multistep.apply(_feat, Q_feat)).detach()
        # mean_adj, scale_adj, prob_adj = self.get_deform_mlp.forward(_feat)
        # probs = torch.stack([prob, prob_adj], dim=-1)
        # probs = torch.softmax(probs, dim=-1)

        # grid_scaling = (STE_multistep.apply(_scaling, Q_scaling)).detach()
        # offsets = (STE_multistep.apply(_grid_offsets, Q_offsets.unsqueeze(1))).detach()
        # offsets = offsets.view(-1, 3*self.n_offsets)
        # mask_tmp = _mask.repeat(1, 1, 3).view(-1, 3*self.n_offsets)

        # bit_feat = self.EG_mix_prob_2.forward(_feat,
        #                                     mean, mean_adj,
        #                                     scale, scale_adj,
        #                                     probs[..., 0], probs[..., 1],
        #                                     Q=Q_feat)

        # bit_scaling = self.entropy_gaussian.forward(grid_scaling, mean_scaling, scale_scaling, Q_scaling)
        # bit_offsets = self.entropy_gaussian.forward(offsets, mean_offsets, scale_offsets, Q_offsets)
        # bit_offsets = bit_offsets * mask_tmp

        # bit_anchor = _anchor.shape[0]*3*anchor_round_digits
        # bit_feat = torch.sum(bit_feat).item()
        # bit_scaling = torch.sum(bit_scaling).item()
        # bit_offsets = torch.sum(bit_offsets).item()
        # if self.ste_binary:
        #     bit_hash = get_binary_vxl_size((hash_embeddings+1)/2)[1].item()
        # else:
        #     bit_hash = hash_embeddings.numel()*32
        # bit_masks = get_binary_vxl_size(_mask)[1].item()

        # print(bit_anchor, bit_feat, bit_scaling, bit_offsets, bit_hash, bit_masks)

        # log_info = f"\nEstimated sizes in MB: " \
        #            f"anchor {round(bit_anchor/bit2MB_scale, 4)}, " \
        #            f"feat {round(bit_feat/bit2MB_scale, 4)}, " \
        #            f"scaling {round(bit_scaling/bit2MB_scale, 4)}, " \
        #            f"offsets {round(bit_offsets/bit2MB_scale, 4)}, " \
        #            f"hash {round(bit_hash/bit2MB_scale, 4)}, " \
        #            f"masks {round(bit_masks/bit2MB_scale, 4)}, " \
        #            f"MLPs {round(self.get_mlp_size()[0]/bit2MB_scale, 4)}, " \
        #            f"Total {round((bit_anchor + bit_feat + bit_scaling + bit_offsets + bit_hash + bit_masks + self.get_mlp_size()[0])/bit2MB_scale, 4)}"

        # return log_info

        Q_feat = 1
        Q_scaling = 0.001
        Q_offsets = 0.2

        mask_anchor = self.get_mask_anchor.to(torch.bool)[:, 0]  # N

        _anchor = self.get_anchor[mask_anchor]
        _feat = self._anchor_feat[mask_anchor]
        _grid_offsets = self._offset[mask_anchor]
        _scaling = self.get_scaling[mask_anchor]
        _mask = self.get_mask[mask_anchor]
        # _sp_anchor_sign = self.spatial_anchor_sign[mask_anchor]2
        
        
        hyper_prior = self.query_triplane(_anchor, training=False)

        # feat_context = self.calc_interp_feat(_anchor)  # [N_visible_anchor*0.2, 32]
        # mean, scale, prob, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
        #     torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[self.feat_dim, self.feat_dim, self.feat_dim, 6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1, 1], dim=-1)  # [N_visible_anchor, 32], [N_visible_anchor, 32]
        # feat_hyper, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
        #     torch.split(self.get_grid_mlp(feat_context), split_size_or_sections=[self.feat_dim*2, 6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1, 1], dim=-1)  # [N_visible_anchor, 32], [N_visible_anchor, 32]
       
        # feat_hyper, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
        #     torch.split(self.query_triplane(_anchor), split_size_or_sections=[self.feat_dim*2, 6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1, 1], dim=-1)  # [N_visible_anchor, 32], [N_visible_anchor, 32]

        mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_scaling_adj, Q_offsets_adj = \
            torch.split(self.mlp_VM(hyper_prior), split_size_or_sections=[6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1], dim=-1)  # [N_visible_anchor, 32], [N_visible_anchor, 32]
        Q_feat_adj = self.mlp_feat_Q(hyper_prior)
        
        # feat_context_orig = self.calc_interp_feat(_anchor)
        # feat_context = self.get_grid_mlp(feat_context_orig)
        # plane_context = self.query_triplane(_anchor)
        # feat_hyper, Q_feat_adj = torch.split(feat_context, split_size_or_sections=[self.feat_dim*2, 1], dim=-1)
        # mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_scaling_adj, Q_offsets_adj = \
        #     torch.split(plane_context, split_size_or_sections=[6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1], dim=-1)

        Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
        Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
        Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))

        # mean_adj, scale_adj, prob_adj = self.get_deform_mlp.forward(_feat)
        # mean_adj, scale_adj, prob_adj = self.get_deform_mlp.forward(_feat, torch.cat([mean, scale, prob], dim=-1).contiguous())

        # ch_ctx = self.get_deform_mlp.forward(_feat)

        _feat = (STE_multistep.apply(_feat, Q_feat)).detach()
        grid_scaling = (STE_multistep.apply(_scaling, Q_scaling)).detach()
        offsets = (STE_multistep.apply(_grid_offsets, Q_offsets.unsqueeze(1))).detach()
        offsets = offsets.view(-1, 3*self.n_offsets) 
        mask_tmp = _mask.repeat(1, 1, 3).view(-1, 3*self.n_offsets)


        # nac_valid_mask = mask_anchor[~self.spatial_anchor_sign]
        # knn_indices = self.knn_indices[nac_valid_mask]

        # nac_sp_feat = self._anchor_feat[knn_indices]
        # mean_sp, scale_sp, prob_sp = self.get_spatial_mlp.forward(nac_sp_feat)
        # bit_feat = torch.empty_like(_feat)
        # probs_ac = torch.stack([prob[_sp_anchor_sign], prob_adj[_sp_anchor_sign]], dim=-1)
        # probs_ac = torch.softmax(probs_ac, dim=-1)

        # probs_nac = torch.stack([prob[~_sp_anchor_sign], prob_adj[~_sp_anchor_sign], prob_sp], dim=-1)
        # probs_nac = torch.softmax(probs_nac, dim=-1)


        # bit_feat[_sp_anchor_sign] = self.EG_mix_prob_2.forward(_feat[_sp_anchor_sign],
        #                                     mean[_sp_anchor_sign], mean_adj[_sp_anchor_sign],
        #                                     scale[_sp_anchor_sign], scale_adj[_sp_anchor_sign],
        #                                     probs_ac[..., 0], probs_ac[..., 1],
        #                                     Q=Q_feat[_sp_anchor_sign])
        
        # bit_feat[~_sp_anchor_sign] = self.EG_mix_prob_3.forward(_feat[~_sp_anchor_sign],
        #                                     mean[~_sp_anchor_sign], mean_adj[~_sp_anchor_sign], mean_sp,
        #                                     scale[~_sp_anchor_sign], scale_adj[~_sp_anchor_sign], scale_sp,
        #                                     probs_nac[..., 0], probs_nac[..., 1], probs_nac[..., 2],
        #                                     Q=Q_feat[~_sp_anchor_sign])

        
        # knn_indices = self.knn_indices[mask_anchor]
        # sp_feat = self._anchor_feat[knn_indices]
        # # mean_sp, scale_sp, prob_sp = self.get_spatial_mlp.forward(sp_feat)
        # mean_sp, scale_sp, prob_sp = self.get_spatial_mlp.forward(sp_feat, torch.cat([mean, scale, prob], dim=-1).contiguous())
        # probs = torch.stack([prob, prob_adj, prob_sp], dim=-1)
        # probs = torch.softmax(probs, dim=-1)
        # bit_feat = self.EG_mix_prob_3.forward(_feat,
        #                                     mean, mean_adj, mean_sp,
        #                                     scale, scale_adj, scale_sp,
        #                                     probs[..., 0], probs[..., 1], probs[..., 2],
        #                                     Q=Q_feat)


        # nac_valid_mask = mask_anchor[~self.spatial_anchor_sign]
        # knn_indices = self.knn_indices[nac_valid_mask]
        # nac_sp_feat = self._anchor_feat[knn_indices]
        # sp_ctx = torch.zeros((_feat.shape[0], 50*2), device='cuda', dtype=torch.float32)
        # sp_ctx[~_sp_anchor_sign] = self.get_spatial_mlp.forward(nac_sp_feat)
        # ac_coords = _anchor[_sp_anchor_sign]
        # ac_sp_ctx_ori = self.calc_interp_feat(ac_coords, ac_sp=True)
        # sp_ctx[_sp_anchor_sign] = self.get_ac_sp_mlp(ac_sp_ctx_ori)

        # knn_indices = self.knn_indices[mask_anchor]
        # sp_feat = self._anchor_feat[knn_indices]
        # sp_ctx = torch.zeros((_feat.shape[0], 50), device='cuda', dtype=torch.float32)
        # # sp_ctx[~_sp_anchor_sign] = self.get_spatial_mlp.forward(sp_feat[~_sp_anchor_sign])
        # ac_sp_ctx_side = self.get_deform_mlp.forward(sp_feat[_sp_anchor_sign].reshape(-1, 50))
        # sp_ctx[_sp_anchor_sign] = self.get_spatial_mlp.forward(ac_sp_ctx_side.reshape(-1, 3, 50))

        # nac_sp_ctx_side = self.get_deform_mlp.forward(sp_feat[~_sp_anchor_sign].reshape(-1, 50))
        # sp_ctx[~_sp_anchor_sign] = self.get_spatial_mlp.forward(nac_sp_ctx_side.reshape(-1, 3, 50))

        # mean, scale = self.get_param_mlp(hyper_prior, ch_ctx, sp_ctx)

        # mean, scale = self.get_param_mlp(hyper_prior, ch_ctx)
        # bit_feat = self.entropy_gaussian(_feat, mean, scale, Q_feat)



        # knn_indices = self.knn_indices[mask_anchor]
        # sp_feat = self._anchor_feat[mask_anchor][knn_indices]
        # sp_ctx = self.get_spatial_mlp.forward(sp_feat)
        # # base_indices = self.base_mask[mask_anchor]
        # # base_pts = _anchor[base_indices]
        # # sp_ctx[base_indices, ...] = self.mlp_base_sp(base_pts)
        # means, scales = self.get_param_mlp(feat_hyper, ch_ctx, sp_ctx)
        # bit_feat = self.entropy_gaussian(_feat, means, scales, Q_feat)


        # choose_base_idx = self.level_0_mask[mask_anchor]
        # non_base_choose_idx = mask_anchor & (~self.level_0_mask)
        # knn_indices = self.knn_indices[non_base_choose_idx]
        # sp_feat = self._anchor_feat[mask_anchor][knn_indices]

        # sp_ctx = torch.zeros((_feat.shape[0], 50*2), device='cuda', dtype=torch.float32)
        # sp_ctx[~choose_base_idx, ...] = self.get_spatial_mlp.forward(sp_feat)

        # # base_pts = _anchor[choose_base_idx]
        # # sp_ctx[choose_base_idx, ...] = self.mlp_base_sp(base_pts)
        # mask_list = [getattr(self, f"level_{i}_mask") for i in range(self.level_num)]
        # means, scales = self.get_param_mlp(hyper_prior, ch_ctx, sp_ctx, mask_list, mask_anchor)
        # bit_feat = self.entropy_gaussian(_feat, means, scales, Q_feat)

        mask_list = [getattr(self, f"level_{i}_mask") for i in range(self.level_num)]
        knn_indices = self.knn_indices[mask_anchor]
        sp_feat = _feat[self.morton_indices][knn_indices]
        
        # sp_ctx = self.get_spatial_mlp.forward(sp_feat)
        sp_ctx = self.get_spatial_mlp.forward(sp_feat, mask_list, mask_anchor)
        ch_ctx = self.get_deform_mlp.forward(_feat, mask_list, mask_anchor)
        # hyper_prior = self.calc_interp_feat(_anchor)
        means, scales = self.get_param_mlp(hyper_prior, ch_ctx, sp_ctx, mask_list, mask_anchor)
        # means, scales = self.get_param_mlp(hyper_prior, ch_ctx, sp_ctx)
        bit_feat = self.entropy_gaussian(_feat, means, scales, Q_feat)


        # knn_indices = self.knn_indices[mask_anchor]
        # sp_feat = self._anchor_feat[knn_indices]
        # sp_ctx = self.get_spatial_mlp.forward(sp_feat)
        # means, scales = self.get_param_mlp(hyper_prior, ch_ctx, sp_ctx)
        # bit_feat = self.entropy_gaussian(_feat, means, scales, Q_feat, self._anchor_feat.mean())

        bit_scaling = self.entropy_gaussian.forward(grid_scaling, mean_scaling, scale_scaling, Q_scaling)
        bit_offsets = self.entropy_gaussian.forward(offsets, mean_offsets, scale_offsets, Q_offsets)
        bit_offsets = bit_offsets * mask_tmp

        bit_anchor = _anchor.shape[0]*3*anchor_round_digits
        bit_feat = torch.sum(bit_feat).item()
        bit_scaling = torch.sum(bit_scaling).item()
        bit_offsets = torch.sum(bit_offsets).item()
        
        # hash_embeddings = self.get_encoding_params()
        # if self.ste_binary:
        #     bit_hash = get_binary_vxl_size((hash_embeddings+1)/2)[1].item()
        # else:
        #     bit_hash = hash_embeddings.numel()*32
        bit_hash = 0

        bit_masks = get_binary_vxl_size(_mask)[1].item()


        print(bit_anchor, bit_feat, bit_scaling, bit_offsets, bit_hash, bit_masks)

        log_info = f"\nEstimated sizes in MB: " \
                   f"anchor {round(bit_anchor/bit2MB_scale, 4)}, " \
                   f"feat {round(bit_feat/bit2MB_scale, 4)}, " \
                   f"scaling {round(bit_scaling/bit2MB_scale, 4)}, " \
                   f"offsets {round(bit_offsets/bit2MB_scale, 4)}, " \
                   f"hash {round(bit_hash/bit2MB_scale, 4)}, " \
                   f"masks {round(bit_masks/bit2MB_scale, 4)}, " \
                   f"MLPs {round(self.get_mlp_size()[0]/bit2MB_scale, 4)}, " \
                   f"Total {round((bit_anchor + bit_feat + bit_scaling + bit_offsets + bit_hash + bit_masks + self.get_mlp_size()[0])/bit2MB_scale, 4)}"

        return log_info

    @torch.no_grad()
    def conduct_encoding(self, pre_path_name):

        t_total = 0
        t_anchor = 0
        t_feat = 0
        t_scaling = 0
        t_offset = 0
        t_mask = 0
        t_codec = 0

        t_total_0 = get_time()
        torch.cuda.synchronize(); t1 = time.time()
        print('Start encoding ...')

        mask_anchor = self.get_mask_anchor.to(torch.bool)[:, 0]  # N

        _anchor = self.get_anchor[mask_anchor]
        _feat = self._anchor_feat[mask_anchor]  # N, 50
        _grid_offsets = self._offset[mask_anchor]  # N, 10, 3
        _scaling = self.get_scaling[mask_anchor]  # N, 6
        _mask = self.get_mask[mask_anchor]  # N, 10, 1
        knn_indices = self.knn_indices[mask_anchor]  # for mortan sorted feat

        N = _anchor.shape[0]

        t_anchor_0 = get_time()
        _anchor_int = torch.round(_anchor / self.voxel_size)
        sorted_indices = calculate_morton_order(_anchor_int)
        _anchor_int = _anchor_int[sorted_indices]
        npz_path= os.path.join(pre_path_name, 'xyz_gpcc.npz')
        means_strings = compress_gpcc(_anchor_int)
        np.savez_compressed(npz_path, voxel_size=self.voxel_size, means_strings=means_strings)
        bits_xyz = os.path.getsize(npz_path) * 8
        t_anchor += get_time() - t_anchor_0

        _anchor = _anchor_int * self.voxel_size
        _feat = _feat[sorted_indices]
        _grid_offsets = _grid_offsets[sorted_indices]
        _scaling = _scaling[sorted_indices]
        _mask = _mask[sorted_indices]
        knn_indices = knn_indices[sorted_indices]

        level_signs, level_indices = self.get_level_indices(_anchor_int, decoding=True)
        encode_order = torch.cat(level_indices)
        dims = [5, 10, 15, 20]

        torch.save(self.x_bound_min, os.path.join(pre_path_name, 'x_bound_min.pkl'))
        torch.save(self.x_bound_max, os.path.join(pre_path_name, 'x_bound_max.pkl'))


        bit_feat_list = []

        masks_b_name = os.path.join(pre_path_name, 'masks.b')

        Q_feat = 1
        Q_scaling = 0.001
        Q_offsets = 0.2



        scaling_b_name = os.path.join(pre_path_name, 'scaling.b')
        offsets_b_name = os.path.join(pre_path_name, 'offsets.b')

        
        hyper_prior = self.query_triplane(_anchor, training=False)
        # encode scaling and offsets
        mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_scaling_adj, Q_offsets_adj = \
            torch.split(self.mlp_VM(hyper_prior), split_size_or_sections=[6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1], dim=-1)  # [N_visible_anchor, 32], [N_visible_anchor, 32]
        Q_feat_adj = self.mlp_feat_Q(hyper_prior)

        ## sort
        _scaling = _scaling[encode_order]
        mean_scaling = mean_scaling[encode_order]
        scale_scaling = scale_scaling[encode_order]
        Q_scaling_adj = Q_scaling_adj[encode_order]

        _mask = _mask[encode_order]
        _grid_offsets = _grid_offsets[encode_order]
        mean_offsets = mean_offsets[encode_order]
        scale_offsets = scale_offsets[encode_order]
        Q_offsets_adj = Q_offsets_adj[encode_order]

        Q_feat_adj = Q_feat_adj.contiguous().repeat(1, _feat.shape[-1])
        Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
        Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)
        mean_scaling = mean_scaling.contiguous().view(-1)
        mean_offsets = mean_offsets.contiguous().view(-1)
        # scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-9)
        scale_scaling = scale_scaling.contiguous().view(-1)

        scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-9)
        Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
        Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
        Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))



        t_scaling_0 = get_time()
        scaling = _scaling.view(-1)  # [N_num*6]
        # scaling = STE_multistep.apply(scaling, Q_scaling, self.get_scaling.mean())
        scaling = STE_multistep.apply(scaling, Q_scaling)
        torch.cuda.synchronize(); t0 = time.time()
        bit_scaling = encoder_gaussian_chunk(scaling, mean_scaling, scale_scaling, Q_scaling, file_name=scaling_b_name, chunk_size=10_0000)
        torch.cuda.synchronize(); t_codec += time.time() - t0
        t_scaling += get_time() - t_scaling_0

        t_offset_0 = get_time()
        mask = _mask.repeat(1, 1, 3).view(-1, 3*self.n_offsets).view(-1).to(torch.bool)  # [N_num*K*3]
        offsets = _grid_offsets.view(-1, 3*self.n_offsets).view(-1)  # [N_num*K*3]
        offsets = STE_multistep.apply(offsets, Q_offsets, self._offset.mean())
        offsets[~mask] = 0.0
        torch.cuda.synchronize(); t0 = time.time()
        bit_offsets = encoder_gaussian_chunk(offsets[mask], mean_offsets[mask], scale_offsets[mask], Q_offsets[mask], file_name=offsets_b_name, chunk_size=10_0000)
        torch.cuda.synchronize(); t_codec += time.time() - t0
        t_offset += get_time() - t_offset_0

        torch.cuda.empty_cache()


        # encode feat
        t_feat_0 = get_time()
        # _feat = STE_multistep.apply(_feat, Q_feat, self._anchor_feat.mean())
        _feat = STE_multistep.apply(_feat, Q_feat)
        sp_feat = _feat[knn_indices]
        sp_ctx = self.get_spatial_mlp.forward(sp_feat, level_signs)
        ch_ctx = self.get_deform_mlp.forward(_feat, level_signs)
        means, scales = self.get_param_mlp(hyper_prior, ch_ctx, sp_ctx, level_signs)
        # scales = torch.clamp(scales, min=1e-9)
        t_feat += get_time() - t_feat_0

        for level_id, level_sign in enumerate(level_signs):
            level_feat = _feat[level_sign]
            level_means = means[level_sign]
            level_scales = scales[level_sign]
            level_Q = Q_feat[level_sign]
            feat_b_name = os.path.join(pre_path_name, 'feat.b').replace('.b', f'_{level_id}.b')
            pre_dims = 0
            bit_feat = 0
            torch.cuda.synchronize(); t0 = time.time()
            t_feat_0 = get_time()
            for dim_id, dim in enumerate(dims):
                level_ch_feat = level_feat[:, pre_dims:pre_dims+dim].contiguous().view(-1)
                level_ch_means = level_means[:, pre_dims:pre_dims+dim].contiguous().view(-1)
                level_ch_scales = level_scales[:, pre_dims:pre_dims+dim].contiguous().view(-1)
                level_ch_Q = level_Q[:, pre_dims:pre_dims+dim].contiguous().view(-1)
                bit_feat += encoder_gaussian_chunk(
                    level_ch_feat,
                    level_ch_means,
                    level_ch_scales,
                    level_ch_Q,
                    file_name=feat_b_name.replace('.b', f'_{dim_id}.b'), chunk_size=50_0000)
                
                pre_dims+=dim

            t_feat += get_time() - t_feat_0
            torch.cuda.synchronize(); t_codec += time.time() - t0
            bit_feat_list.append(bit_feat)

            torch.cuda.empty_cache()



        bit_anchor = bits_xyz
        bit_feat = sum(bit_feat_list)


        t_mask_0 = get_time()
        bit_masks = encoder(_mask, file_name=masks_b_name)
        t_mask += get_time() - t_mask_0

        t_total += get_time() - t_total_0

        torch.cuda.synchronize(); t2 = time.time()
        print('encoding time:', t2 - t1)
        print('codec time:', t_codec)

        # adjust opacity and rotation order for prefilter
        with torch.no_grad():
            self._opacity.data = self._opacity[mask_anchor][sorted_indices][encode_order]
            self._rotation.data = self._rotation[mask_anchor][sorted_indices][encode_order]


        # 32*3*2/bit2MB_scale is for xyz_bound_min and xyz_bound_max
        log_info = f"\nEncoded sizes in MB: " \
                   f"anchor {round(bit_anchor/bit2MB_scale, 4)}, " \
                   f"feat {round(bit_feat/bit2MB_scale, 4)}, " \
                   f"scaling {round(bit_scaling/bit2MB_scale, 4)}, " \
                   f"offsets {round(bit_offsets/bit2MB_scale, 4)}, " \
                   f"masks {round(bit_masks/bit2MB_scale, 4)}, " \
                   f"MLPs {round(self.get_mlp_size()[0]/bit2MB_scale, 4)}, " \
                   f"Total {round((bit_anchor + bit_feat + bit_scaling + bit_offsets + bit_masks + self.get_mlp_size()[0])/bit2MB_scale + 32*3*2/bit2MB_scale, 4)}, " \
                   f"EncTime {round(t2 - t1, 4)}"
        log_info_time = f"\nEncoded time in s: " \
                   f"anchor {round(t_anchor, 4)}, " \
                   f"feat {round(t_feat, 4)}, " \
                   f"scaling {round(t_scaling, 4)}, " \
                   f"offsets {round(t_offset, 4)}, " \
                   f"masks {round(t_mask, 4)}, " \
                   f"Total {round(t_total, 4)}"
        log_info = log_info + log_info_time

        
        return log_info

    @torch.no_grad()
    def conduct_decoding(self, pre_path_name):

        t_total = 0
        t_anchor = 0
        t_feat = 0
        t_scaling = 0
        t_offset = 0
        t_mask = 0

        t_total_0 = get_time()

        torch.cuda.synchronize(); t1 = time.time()
        print('Start decoding ...')

        self.x_bound_min = torch.load(os.path.join(pre_path_name, 'x_bound_min.pkl'))
        self.x_bound_max = torch.load(os.path.join(pre_path_name, 'x_bound_max.pkl'))


        masks_b_name = os.path.join(pre_path_name, 'masks.b')

        t_anchor_0 = get_time()
        npz_path = os.path.join(pre_path_name, 'xyz_gpcc.npz')
        data_dict = np.load(npz_path)
        voxel_size = float(data_dict['voxel_size'])
        means_strings = data_dict['means_strings'].tobytes()
        _anchor_int_dec = decompress_gpcc(means_strings).to('cuda')
        sorted_indices = calculate_morton_order(_anchor_int_dec)
        _anchor_int_dec = _anchor_int_dec[sorted_indices]
        anchor_decoded = _anchor_int_dec * voxel_size
        t_anchor += get_time() - t_anchor_0
        N = anchor_decoded.shape[0]

        level_signs, level_indices = self.get_level_indices(_anchor_int_dec, decoding=True)
        encode_order = torch.cat(level_indices)
        anchor_decoded = anchor_decoded[encode_order]
        level_nums = [indice.shape[0] for indice in level_indices]
        dims = [5, 10, 15, 20]

        t_mask_0 = get_time()
        masks_decoded = decoder(N*self.n_offsets, masks_b_name)  # {0, 1}
        masks_decoded = masks_decoded.view(-1, self.n_offsets, 1)
        t_mask += get_time() - t_mask_0


        scaling_b_name = os.path.join(pre_path_name, 'scaling.b')
        offsets_b_name = os.path.join(pre_path_name, 'offsets.b')

        Q_feat = 1
        Q_scaling = 0.001
        Q_offsets = 0.2

        hyper_prior = self.query_triplane(anchor_decoded, training=False)
        # decode scaling and offsets
        mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_scaling_adj, Q_offsets_adj = \
            torch.split(self.mlp_VM(hyper_prior), split_size_or_sections=[6, 6, 3*self.n_offsets, 3*self.n_offsets, 1, 1], dim=-1)  # [N_visible_anchor, 32], [N_visible_anchor, 32]
        Q_feat_adj = self.mlp_feat_Q(hyper_prior)

        Q_feat_adj = Q_feat_adj.contiguous().repeat(1, 50)
        Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1]).view(-1)
        Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1]).view(-1)

        mean_scaling = mean_scaling.contiguous().view(-1)
        mean_offsets = mean_offsets.contiguous().view(-1)

        # scale_scaling = torch.clamp(scale_scaling.contiguous().view(-1), min=1e-9)
        scale_scaling = scale_scaling.contiguous().view(-1)
        
        scale_offsets = torch.clamp(scale_offsets.contiguous().view(-1), min=1e-9)
        Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
        Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
        Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj))

        t_scaling_0 = get_time()
        scaling_decoded = decoder_gaussian_chunk(mean_scaling, scale_scaling, Q_scaling, file_name=scaling_b_name, chunk_size=10_0000)
        scaling_decoded = scaling_decoded.view(N, 6)  # [N_num, 6]
        t_scaling += get_time() - t_scaling_0

        t_offset_0 = get_time()
        masks_tmp = masks_decoded.repeat(1, 1, 3).view(-1, 3 * self.n_offsets).view(-1).to(torch.bool)
        offsets_decoded_tmp = decoder_gaussian_chunk(mean_offsets[masks_tmp], scale_offsets[masks_tmp], Q_offsets[masks_tmp], file_name=offsets_b_name, chunk_size=10_0000)
        offsets_decoded = torch.zeros_like(mean_offsets)
        offsets_decoded[masks_tmp] = offsets_decoded_tmp
        offsets_decoded = offsets_decoded.view(N, -1).view(N, self.n_offsets, 3)  # [N_num, K, 3]
        t_offset += get_time() - t_offset_0

        torch.cuda.empty_cache()



        # decode feat
        knn_indices_list = self.decoding_knn(anchor_decoded, level_nums)
        feat_decoded = torch.empty([0, 50], dtype=torch.float32, device='cuda') 
        pre_nums = 0
        for level_id, level_num in enumerate(level_nums):
            feat_b_name = os.path.join(pre_path_name, 'feat.b').replace('.b', f'_{level_id}.b')
            pre_dims = 0
            tmp_decoded = torch.empty([level_num, 0], dtype=torch.float32, device='cuda')
            tmp_hyper = hyper_prior[pre_nums:pre_nums+level_num]
            tmp_Q = Q_feat[pre_nums:pre_nums+level_num]
            t_feat_0 = get_time()
            for dim_id, dim in enumerate(dims):
                ch_ctx = self.get_deform_mlp.forward(feat_q=tmp_decoded, ch_to_dec=dim_id, decoding=True, group_to_dec=level_id, group_num=level_num)
                if level_id == 0:
                    means, scales = self.get_param_mlp(hyper_prior=tmp_hyper, ch_ctx=ch_ctx, ch_to_dec=dim_id, decoding=True, group_to_dec=level_id)
                else:
                    knn_indices = knn_indices_list[level_id-1]
                    sp_feat = feat_decoded[:, pre_dims:pre_dims+dim][knn_indices]
                    sp_ctx = self.get_spatial_mlp(feat_q=sp_feat, ch_to_dec=dim_id, decoding=True, group_to_dec=level_id)
                    means, scales = self.get_param_mlp(hyper_prior=tmp_hyper, ch_ctx=ch_ctx, sp_ctx=sp_ctx, ch_to_dec=dim_id, decoding=True, group_to_dec=level_id)

                # scales = torch.clamp(scales, min=1e-9)

                ch_decoded = decoder_gaussian_chunk(
                    means.contiguous().view(-1),
                    scales.contiguous().view(-1),
                    tmp_Q[:, pre_dims:pre_dims+dim].contiguous().view(-1),
                    file_name=feat_b_name.replace('.b', f'_{dim_id}.b'), 
                    chunk_size=50_0000
                )
                ch_decoded = ch_decoded.view(-1, dim).cuda()
                pre_dims+=dim
                tmp_decoded = torch.cat([tmp_decoded, ch_decoded], dim=-1)

            pre_nums+=level_num
            feat_decoded = torch.cat([feat_decoded, tmp_decoded], dim=0)
            t_feat += get_time() - t_feat_0

        torch.cuda.empty_cache()

        t_total += get_time() - t_total_0

        torch.cuda.synchronize(); t2 = time.time()
        print('decoding time:', t2 - t1)

        # fill back N_full
        _anchor = torch.zeros(size=[N, 3], device='cuda')
    
        _offset = torch.zeros(size=[N, self.n_offsets, 3], device='cuda')
        _scaling = torch.zeros(size=[N, 6], device='cuda')
        _mask = torch.zeros(size=[N, self.n_offsets+1, 1], device='cuda')

        _anchor[:N] = anchor_decoded

        _offset[:N] = offsets_decoded
        _scaling[:N] = scaling_decoded
        _mask[:N, :10] = masks_decoded

        print('Start replacing parameters with decoded ones...')
        # replace attributes by decoded ones
        self._anchor_feat = nn.Parameter(feat_decoded)
        self._offset = nn.Parameter(_offset)
        self.decoded_version = True
        self._anchor = nn.Parameter(_anchor)
        self._scaling = nn.Parameter(_scaling)
        self._mask = nn.Parameter(_mask)
        


        log_info = f"\nDecTime {round(t2 - t1, 4)}"

        log_info_time = f"\nDecoded time in s: " \
                        f"anchor {round(t_anchor, 4)}, " \
                        f"feat {round(t_feat, 4)}, " \
                        f"scaling {round(t_scaling, 4)}, " \
                        f"offsets {round(t_offset, 4)}, " \
                        f"masks {round(t_mask, 4)}, " \
                        f"Total {round(t_total, 4)}"
        log_info = log_info + log_info_time

        return log_info


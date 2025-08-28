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
import os.path
import time

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from einops import repeat

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.encodings import STE_binary, STE_multistep, QuantizeWithGrad
from compressai.ops import quantize_ste

def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False, step=0):
    ## view frustum filtering for acceleration

    time_sub = 0

    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)

    anchor = pc.get_anchor[visible_mask]
    feat = pc._anchor_feat[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]
    binary_grid_masks = pc.get_mask[visible_mask]  # [N_vis, 10, 1] 
    mask_anchor = pc.get_mask_anchor[visible_mask]

    # bit_per_ac_feat = None
    # bit_per_nac_feat = None

    bit_per_param = None
    bit_per_feat_param = None
    bit_per_scaling_param = None
    bit_per_offsets_param = None
    bit_per_knn = None
    dis_loss = None

    bit_feat_levels = []
    Q_feat = 1
    Q_scaling = 0.001
    Q_offsets = 0.2

    epsilon = 1e-9

    if is_training:
        if step > 3000 and step <= 10000:
            # quantization
            feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat
            grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling
            grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets

        if step == 10000:
            pc.update_anchor_bound()
            # pc.encoding_xyz_config()
            # pc.sp_config()
            # pc.ch_config()
            pc.param_config()
            # pc.VM_config()
            pc.triplane_config()
            # pc.mlp_knn_config()
            pc.feat_q_config()
            # pc.feat_weight_config()
            # pc.feat_mask_config()

        # if step ==15_000:


        if step > 10000:
            

            # for rendering
            # feat_context_orig = pc.calc_interp_feat(anchor)
            # feat_context = pc.get_grid_mlp(feat_context_orig)
            # mean, scale, prob, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
            #     torch.split(feat_context, split_size_or_sections=[pc.feat_dim, pc.feat_dim, pc.feat_dim, 6, 6, 3*pc.n_offsets, 3*pc.n_offsets, 1, 1, 1], dim=-1)
            hyper_prior = pc.query_triplane(anchor)


            # plane_context = pc.query_triplane(anchor)
            

            # feat_hyper, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
            #     torch.split(hyper_prior, split_size_or_sections=[pc.feat_dim*2, 6, 6, 3*pc.n_offsets, 3*pc.n_offsets, 1, 1, 1], dim=-1)
            
            # feat_hyper, Q_feat_adj = torch.split(feat_context, split_size_or_sections=[pc.feat_dim*2, 1], dim=-1)
            # mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_scaling_adj, Q_offsets_adj = \
            #     torch.split(plane_context, split_size_or_sections=[6, 6, 3*pc.n_offsets, 3*pc.n_offsets, 1, 1], dim=-1)

            mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_scaling_adj, Q_offsets_adj = \
                torch.split(pc.mlp_VM(hyper_prior), split_size_or_sections=[6, 6, 3*pc.n_offsets, 3*pc.n_offsets, 1, 1], dim=-1)
            
            # with torch.no_grad():
            #     anchor_int = pc.voxelize(anchor)
            #     level_signs, _ = pc.get_level_indices(anchor_int, decoding=True)
            # Q_feat_adj = pc.mlp_feat_Q.forward(hyper_prior, level_signs)
            
            
            Q_feat_adj = pc.mlp_feat_Q(hyper_prior)
            # Q_feat_adj = pc.mlp_feat_Q(anchor)

            # Q_feat_adj_base = Q_feat_adj[..., :pc.base_feat_dim]
            # Q_feat_adj_bias = Q_feat_adj[..., pc.base_feat_dim:].repeat(1, pc.n_offsets)
            # Q_feat_adj = torch.cat([Q_feat_adj_base, Q_feat_adj_bias], dim=1).contiguous()


            Q_feat_adj = Q_feat_adj.contiguous()
            Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1])
            Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1])

            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))+epsilon
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj)).view(-1, pc.n_offsets, 3)
            feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat
            # feat = QuantizeWithGrad.apply(feat, Q_feat)
            # feat = STE_multistep.apply(feat, Q_feat)
            # feat = quantize_ste(feat/Q_feat)*Q_feat

            grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling
            grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets
            grid_offsets = grid_offsets.view(-1, 3 * pc.n_offsets)

            # if step % 10 ==0:
            #     print("visible num:", anchor.shape[0])
            #     anchor_coords = pc.spatial_coords[pc.spatial_anchor_sign]
            #     nonanchor_coords = pc.spatial_coords[~pc.spatial_anchor_sign]
            #     # anchor_coords = pc.get_anchor[pc.spatial_anchor_sign]
            #     # nonanchor_coords = pc.get_anchor[~pc.spatial_anchor_sign]
            #     anchor_per_nonanchor, average_per_kernel = pc.spatial_efficiency(anchor_coords, nonanchor_coords)
            #     print(torch.min(anchor_per_nonanchor))
            #     print("spatial_efficiency:", average_per_kernel) 

            # for entropy   
            
            # # visible trick
            # # choose_idx = torch.rand_like(pc.get_anchor[:, 0]) <= 0.2
            # # choose_idx =  choose_idx & pc.get_mask_anchor.to(torch.bool)[:, 0]
            # # choose_idx = choose_idx & visible_mask
            # choose_idx = visible_mask&pc.get_mask_anchor.to(torch.bool)[:, 0]
            # visible_choose_idx = choose_idx[visible_mask]
            # anchor_chosen = anchor[visible_choose_idx]
            # feat_chosen = feat[visible_choose_idx]
            # grid_offsets_chosen = grid_offsets[visible_choose_idx]
            # grid_scaling_chosen = grid_scaling[visible_choose_idx]
            # binary_grid_masks_chosen = binary_grid_masks[visible_choose_idx]
            # mask_anchor_chosen = mask_anchor[visible_choose_idx]
            # hyper_prior_chosen = hyper_prior[visible_choose_idx]
            # Q_feat_chosen = Q_feat[visible_choose_idx]

            choose_idx = torch.rand_like(pc.get_anchor[:, 0]) <= 0.05
            choose_idx =  choose_idx & pc.get_mask_anchor.to(torch.bool)[:, 0]

            anchor_chosen = pc.get_anchor[choose_idx]
            feat_chosen = pc._anchor_feat[choose_idx]
            grid_offsets_chosen = pc._offset[choose_idx]
            grid_scaling_chosen = pc.get_scaling[choose_idx]
            binary_grid_masks_chosen = pc.get_mask[choose_idx]  # [N, 10, 1]
            mask_anchor_chosen = pc.get_mask_anchor[choose_idx]  # [N, 1]


            # feat_context_orig = pc.calc_interp_feat(anchor_chosen)
            # feat_context = pc.get_grid_mlp(feat_context_orig)

            hyper_prior = pc.query_triplane(anchor_chosen)     # ori

            # mean, scale, prob, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
            #     torch.split(feat_context, split_size_or_sections=[pc.feat_dim, pc.feat_dim, pc.feat_dim, 6, 6, 3*pc.n_offsets, 3*pc.n_offsets, 1, 1, 1], dim=-1)

            # plane_context = pc.query_triplane(anchor_chosen)
            # feat_hyper, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
            #     torch.split(feat_context, split_size_or_sections=[pc.feat_dim*2, 6, 6, 3*pc.n_offsets, 3*pc.n_offsets, 1, 1, 1], dim=-1)

            # feat_hyper, Q_feat_adj = torch.split(feat_context, split_size_or_sections=[pc.feat_dim*2, 1], dim=-1)
            # mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_scaling_adj, Q_offsets_adj = \
            #     torch.split(plane_context, split_size_or_sections=[6, 6, 3*pc.n_offsets, 3*pc.n_offsets, 1, 1], dim=-1)        

            mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_scaling_adj, Q_offsets_adj = \
                torch.split(pc.mlp_VM(hyper_prior), split_size_or_sections=[6, 6, 3*pc.n_offsets, 3*pc.n_offsets, 1, 1], dim=-1)      # ori
            
            # # visible trick
            # mean_scaling = mean_scaling[visible_choose_idx]
            # scale_scaling = scale_scaling[visible_choose_idx]
            # mean_offsets = mean_offsets[visible_choose_idx]
            # scale_offsets = scale_offsets[visible_choose_idx]
            # Q_scaling = Q_scaling[visible_choose_idx]
            # Q_offsets = Q_offsets[visible_choose_idx]

            # with torch.no_grad():
            #     anchor_chosen_int = pc.voxelize(anchor_chosen)
            #     level_signs, _ = pc.get_level_indices(anchor_chosen_int, decoding=True)
            # Q_feat_adj = pc.mlp_feat_Q.forward(hyper_prior, level_signs)

            Q_feat_adj = pc.mlp_feat_Q(hyper_prior)       # ori
            # feat_weight = pc.mlp_feat_weight(hyper_prior)   # [N, knn]
            # feat_weight = torch.softmax(feat_weight, dim=1)
            # Q_feat_adj = pc.mlp_feat_Q(anchor_chosen)
            # pc.update_Q_feat_adj(Q_feat_adj, choose_idx)

            # Q_feat_adj_base = Q_feat_adj[..., :pc.base_feat_dim]
            # Q_feat_adj_bias = Q_feat_adj[..., pc.base_feat_dim:].repeat(1, pc.n_offsets)
            # Q_feat_adj = torch.cat([Q_feat_adj_base, Q_feat_adj_bias], dim=1).contiguous()

            Q_feat = 1
            Q_scaling = 0.001
            Q_offsets = 0.2
            Q_feat_adj = Q_feat_adj.contiguous()
            Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1])
            Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1])
            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj)) + epsilon
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj)).view(-1, pc.n_offsets, 3)
            feat_chosen = feat_chosen + torch.empty_like(feat_chosen).uniform_(-0.5, 0.5) * Q_feat        # ori

            # ori
            grid_scaling_chosen = grid_scaling_chosen + torch.empty_like(grid_scaling_chosen).uniform_(-0.5, 0.5) * Q_scaling
            grid_offsets_chosen = grid_offsets_chosen + torch.empty_like(grid_offsets_chosen).uniform_(-0.5, 0.5) * Q_offsets
            grid_offsets_chosen = grid_offsets_chosen.view(-1, 3 * pc.n_offsets)

            # feat_chosen = QuantizeWithGrad.apply(feat_chosen, Q_feat)
            # feat_chosen = STE_multistep.apply(feat_chosen, Q_feat)
            # feat_chosen = quantize_ste(feat_chosen/Q_feat)*Q_feat

            # mean_adj, scale_adj, prob_adj = pc.get_deform_mlp.forward(feat_chosen, torch.cat([mean, scale, prob], dim=-1).contiguous())
            # mean_adj, scale_adj, prob_adj = pc.get_deform_mlp.forward(feat_chosen)

            # ch_ctx = pc.get_deform_mlp.forward(feat_chosen)
            # mean, scale = pc.get_param_mlp(hyper_prior, ch_ctx)
            mean, scale = pc.get_param_mlp(hyper_prior)
            bit_feat = pc.entropy_gaussian(feat_chosen, mean, scale, Q_feat, pc._anchor_feat.mean())

            # sp_anchor_sign = pc.spatial_anchor_sign[choose_idx]  
            # nac_choose_idx = choose_idx[~pc.spatial_anchor_sign]
            # knn_indices = pc.knn_indices[nac_choose_idx]
            # nac_sp_feat = pc._anchor_feat[knn_indices]
            # ac_coords = anchor_chosen[sp_anchor_sign]
            # ac_sp_ctx_side = pc.calc_interp_feat(ac_coords, ac_sp=True)
            # sp_ctx = torch.zeros((feat_chosen.shape[0], 50*2), device='cuda', dtype=torch.float32)
            # sp_ctx[sp_anchor_sign] = pc.get_ac_sp_mlp(ac_sp_ctx_side)
            # sp_ctx[~sp_anchor_sign] = pc.get_spatial_mlp.forward(nac_sp_feat)
            # mean, scale = pc.get_param_mlp(feat_hyper, ch_ctx, sp_ctx)
            # bit_feat = pc.entropy_gaussian(feat_chosen, mean, scale, Q_feat, pc._anchor_feat.mean())

            # knn_indices = pc.knn_indices[choose_idx]
            # sp_feat = pc._anchor_feat[pc.refer_mask][knn_indices]
            # sp_ctx = pc.get_spatial_mlp.forward(sp_feat)
            # # base_indices = pc.base_mask[choose_idx]
            # # base_pts = anchor_chosen[base_indices]
            # # sp_ctx[base_indices, ...] = pc.mlp_base_sp(base_pts)
            # means, scales = pc.get_param_mlp(feat_hyper, ch_ctx, sp_ctx)
            # bit_feat = pc.entropy_gaussian(feat_chosen, means, scales, Q_feat, pc._anchor_feat.mean())

            # choose_base_idx = pc.level_0_mask[choose_idx]
            # non_base_choose_idx = choose_idx & (~pc.level_0_mask)
            # knn_indices = pc.knn_indices[non_base_choose_idx]
            # sp_feat = pc._anchor_feat[pc.refer_mask][knn_indices]
            # sp_ctx = torch.zeros((feat_chosen.shape[0], 50*2), device='cuda', dtype=torch.float32)
            # sp_ctx[~choose_base_idx, ...] = pc.get_spatial_mlp.forward(sp_feat)
            # mask_list = [getattr(pc, f"level_{i}_mask") for i in range(pc.level_num)]
            # means, scales = pc.get_param_mlp(hyper_prior, ch_ctx, sp_ctx, mask_list, choose_idx)
            # bit_feat = pc.entropy_gaussian(feat_chosen, means, scales, Q_feat, pc._anchor_feat.mean())

            # # visible trick
            # mask_list = [getattr(pc, f"level_{i}_mask") for i in range(pc.level_num)]
            # knn_indices = pc.knn_indices[choose_idx]
            # all_indices =  torch.arange(pc.get_anchor.shape[0], device='cuda').long()
            # sp_indices = all_indices[pc.refer_mask][pc.morton_indices][knn_indices]  # (N, knn)
            # unique_sp_indices, inverse_indices  = torch.unique(sp_indices, return_inverse=True, sorted=False)
            # visible_indices = visible_mask.nonzero(as_tuple=True)[0]
            # common_values, idx_in_A_common, idx_in_B_common, A_minus_B, idx_in_A_minus_B = pc.analyze_tensors_vectorized(unique_sp_indices, visible_indices)
            # tmp_sp_feat = torch.zeros((unique_sp_indices.shape[0], 50), dtype=torch.float32).cuda()
            # tmp_sp_feat[idx_in_A_common] = feat[idx_in_B_common]

            # if step%100 ==0:
            #     print("visible num:", visible_indices.shape[0]) 
            #     print("chosen num:", feat_chosen.shape[0])
            #     print("nuique num:", unique_sp_indices.shape[0])
            #     print("common num:", common_values.shape[0]) 
            #     print("new num:", A_minus_B.shape[0]) 


            # new_idx = A_minus_B
            # new_sp_feat = pc._anchor_feat[new_idx]  # (N, feat.shape[-1])
            # new_anchor = pc.get_anchor[new_idx]  # (N*knn, anchor.shape[-1])
            # new_Q_feat_sp_adj = pc.mlp_feat_Q(pc.query_triplane(new_anchor))  #(New, 1)
            # new_Q_feat_sp_adj = new_Q_feat_sp_adj.contiguous().repeat(1, 50)
            # Q_feat_sp = 1
            # new_Q_feat_sp = Q_feat_sp * (1 + torch.tanh(new_Q_feat_sp_adj)) + epsilon
            # # sp_feat = QuantizeWithGrad.apply(sp_feat, Q_feat_sp)
            # new_sp_feat = new_sp_feat + torch.empty_like(new_sp_feat).uniform_(-0.5, 0.5) * new_Q_feat_sp
            # tmp_sp_feat[idx_in_A_minus_B] = new_sp_feat


            # sp_feat = tmp_sp_feat[inverse_indices]  # (N, K ,50)
            # sp_ctx = pc.get_spatial_mlp.forward(sp_feat, mask_list, choose_idx)
            # # sp_ctx = pc.get_spatial_mlp.forward(sp_feat)
            
            # ch_ctx = pc.get_deform_mlp.forward(feat_chosen, mask_list, choose_idx)
            # # hyper_prior = pc.calc_interp_feat(anchor_chosen)
            # means, scales = pc.get_param_mlp(hyper_prior_chosen, ch_ctx, sp_ctx, mask_list, choose_idx)
            # # means, scales = pc.get_param_mlp(hyper_prior, ch_ctx, sp_ctx)
            # bit_feat = pc.entropy_gaussian(feat_chosen, means, scales, Q_feat_chosen, pc._anchor_feat.mean())


            # # ori
            # with torch.no_grad():
            #     mask_list = [getattr(pc, f"level_{i}_mask") for i in range(pc.level_num)]

            #     knn_indices = pc.knn_indices[choose_idx]
            #     sp_feat = pc._anchor_feat[pc.refer_mask][pc.morton_indices][knn_indices]  # (N, knn, feat.shape[-1])

                
            #     # with torch.no_grad():
            #     sp_anchor = pc.get_anchor[pc.refer_mask][pc.morton_indices][knn_indices].view(-1, 3)  # (N*knn, anchor.shape[-1])
            #     # with torch.no_grad():
            #     #     sp_anchor_int = pc.voxelize(sp_anchor)
            #     #     level_signs, _ = pc.get_level_indices(sp_anchor_int, decoding=True)
            #     # Q_feat_sp_adj = pc.mlp_feat_Q.forward(pc.query_triplane(sp_anchor), level_signs).view(-1, pc.knn_num, 1)  #(N, knn, 1)
            #     Q_feat_sp_adj = pc.mlp_feat_Q(pc.query_triplane(sp_anchor)).view(-1, pc.knn_num, 1)  #(N, knn, 1)
            #     # Q_feat_sp_adj = pc.mlp_feat_Q(sp_anchor).view(-1, pc.knn_num, 1)  #(N, knn, 1)  
            #     # Q_feat_sp_adj = pc.tmp_Q_feat_adj[pc.refer_mask][pc.morton_indices][knn_indices]  #(N, knn, 1)
            #     Q_feat_sp_adj = Q_feat_sp_adj.contiguous().repeat(1, 1, 50)
            #     Q_feat_sp = 1
            #     Q_feat_sp = Q_feat_sp * (1 + torch.tanh(Q_feat_sp_adj)) + epsilon
            #     # sp_feat = QuantizeWithGrad.apply(sp_feat, Q_feat_sp)
            #     sp_feat = sp_feat + torch.empty_like(sp_feat).uniform_(-0.5, 0.5) * Q_feat_sp


            # sp_feat = STE_multistep.apply (sp_feat, Q_feat_sp)
            # sp_feat = quantize_ste(sp_feat/Q_feat_sp)*Q_feat_sp

            # sp_feat_mask = pc.get_feat_mask[pc.refer_mask][pc.morton_indices][knn_indices]  # (N, knn, 1)
            # norm = sp_feat_mask.sum(dim=1) + 1e-6  # (N, 1)
            # sp_feat = (sp_feat * sp_feat_mask).sum(dim=1)/norm  # average weighted [N, 50]


            # sp_feat = torch.einsum('nk,nkd->nd', feat_weight, sp_feat)

            # sp_scaling = pc.get_scaling[pc.refer_mask][pc.morton_indices][knn_indices][..., 3:]   # (N, knn, 3)
            # query_scaling = grid_scaling_chosen[:, 3:].unsqueeze(1).repeat(1, pc.knn_num, 1)  # (N, knn, 3)
            # sp_feat = torch.cat([sp_feat, sp_scaling, query_scaling], dim=-1)



            # sp_ctx = pc.get_spatial_mlp.forward(sp_feat, mask_list, choose_idx)
            # ch_ctx = pc.get_deform_mlp.forward(feat_chosen, mask_list, choose_idx)
            # # hyper_prior = pc.calc_interp_feat(anchor_chosen)
            # means, scales = pc.get_param_mlp(hyper_prior, ch_ctx, sp_ctx, mask_list, choose_idx)
            # # means, scales = pc.get_param_mlp(hyper_prior, sp_ctx, mask_list, choose_idx)
            # # means, scales = pc.get_param_mlp(hyper_prior, ch_ctx, sp_ctx)
            # bit_feat = pc.entropy_gaussian(feat_chosen, means, scales, Q_feat, pc._anchor_feat.mean())


                        
                        
            with torch.no_grad():
                if step%1000==0:
                    tmp_mask = torch.sum(binary_grid_masks_chosen[:50].squeeze(), dim=1)
                    tmp_bit = torch.sum(bit_feat[:50], dim=1)
                    print("tmp_mask:", tmp_mask)
                    print("tmp_bit:", tmp_bit)

            # sp_ctx = pc.get_spatial_mlp.forward(sp_feat)
            # ch_ctx = pc.get_deform_mlp.forward(feat_chosen)
            # ch_chunks_1 = torch.split(ch_ctx[0], [d * 2 for d in pc.dims], dim=-1)
            # ch_chunks_2 = torch.split(ch_ctx[1], [d * 2 for d in pc.dims], dim=-1)
            # sp_chunks = torch.split(sp_ctx[0], [d * 2 for d in pc.dims], dim=-1)


            # means_1 = []
            # means_2 = []
            # scales_1 = []
            # scales_2 = []
            # for i, d in enumerate(pc.dims):
            #     ch_1 = ch_chunks_1[i]
            #     ch_2 = ch_chunks_2[i]
            #     sp = sp_chunks[i]
            #     x_1 = torch.cat([hyper_prior, ch_1], dim=-1)
            #     x_2 = torch.cat([hyper_prior, ch_2, sp], dim=-1)
            #     mlp_1 = pc.get_param_mlp.groups[0][i]
            #     mlp_2 = pc.get_param_mlp.groups[1][i]
            #     m_1, s_1 = torch.chunk(mlp_1(x_1), chunks=2, dim=-1)
            #     m_2, s_2 = torch.chunk(mlp_2(x_2), chunks=2, dim=-1)
            #     means_1.append(m_1)
            #     means_2.append(m_2)
            #     scales_1.append(s_1)
            #     scales_2.append(s_2)

            # means_1 = torch.cat(means_1, dim=-1)
            # means_2 = torch.cat(means_2, dim=-1)
            # scales_1 = torch.cat(scales_1, dim=-1)
            # scales_2 = torch.cat(scales_2, dim=-1)

            # bit_feat_1 = pc.entropy_gaussian(feat_chosen, means_1, scales_1, Q_feat, pc._anchor_feat.mean())
            # bit_feat_2 = pc.entropy_gaussian(feat_chosen, means_2, scales_2, Q_feat, pc._anchor_feat.mean())
            
            # bit_feat = bit_feat_1*feat_mask_chosen + bit_feat_2*(1-feat_mask_chosen)

            # with torch.no_grad():
            #     feat_mask_chosen = feat_mask_chosen.to(torch.bool).squeeze()
            #     bit_feat_level_0 = bit_feat[feat_mask_chosen]
            #     bit_feat_level_1 = bit_feat[~feat_mask_chosen]
            #     bit_feat_levels.append(torch.sum(bit_feat_level_0) / bit_feat_level_0.numel())
            #     bit_feat_levels.append(torch.sum(bit_feat_level_1) / bit_feat_level_1.numel())
            #     if step%100==0:
            #         print("level_0 num:", bit_feat_level_0.shape[0])
            #         print("level_1 num:", bit_feat_level_1.shape[0])    

            # # knn_indices compress(hyper_prior, anchor, mortan order | indices)
            # query_mask = mask_list[1]
            # query_mask_chosen = query_mask & choose_idx
            # tmp_mask = query_mask[choose_idx]
            # query_hyper = hyper_prior[tmp_mask]
            # query_anchor = anchor_chosen[tmp_mask]
            # query_order = pc.query_order[query_mask_chosen]
            # refer_knn = pc.refer_knn[query_mask_chosen]
            # feat_knn = pc.feat_knn[query_mask_chosen]
            # refer_anchor = sp_anchor.view(-1, pc.knn_num, 3)[tmp_mask]
            # query_info = torch.cat([query_anchor, query_hyper],dim=-1)
            # knn_means, knn_scales = torch.split(pc.mlp_knn(query_info), split_size_or_sections=[1, 1], dim=-1)
            # bit_knn = pc.entropy_gaussian(feat_knn, knn_means, knn_scales, 1, pc.feat_knn.mean())

            # # knn offsets compress
            # voxel_query_anchor = pc.voxelize(query_anchor)  # [N ,3]
            # voxel_refer_anchor = pc.voxelize(refer_anchor)  # [N, knn, 3]
            # anchor_offsets = (voxel_refer_anchor - voxel_query_anchor.unsqueeze(1)).view(-1, pc.knn_num*3)  #[N, knn*3]
            # context_ori = pc.calc_interp_feat(query_anchor)
            # mean_knn, scale_knn = torch.split(pc.get_grid_mlp(context_ori), split_size_or_sections=[6, 6], dim=-1)
            # bit_knn = pc.entropy_gaussian(anchor_offsets, mean_knn, scale_knn, 1)

            
        

            # print(query_order[:15])
            # print(refer_knn[:15])
            # if step%1000==0:
            #     print(torch.round(query_anchor[:20]/pc.voxel_size))
            #     print(torch.round(refer_anchor[:20]/pc.voxel_size))
                # print(pc.feat_knn[query_mask_chosen][:20])

            # # enhence spatial relation
            # feat_chosen_expanded = feat_chosen.unsqueeze(1)  # [N, 1, 50]
            # # 计算差值并求平方和（L2 距离的平方），再开平方
            # dist = torch.norm(feat_chosen_expanded - sp_feat, dim=-1)  # [N, 2]
            # # 对每一行的两个距离求平均
            # mean_dist = dist.mean(dim=1)  # [N]
            # dis_loss = torch.sum(mean_dist) / mean_dist.numel()



            # knn_indices = pc.knn_indices[choose_idx]
            # sp_feat = pc._anchor_feat[knn_indices]
            # sp_ctx = pc.get_spatial_mlp.forward(sp_feat)
            # means, scales = pc.get_param_mlp(hyper_prior, ch_ctx, sp_ctx)
            # bit_feat = pc.entropy_gaussian(feat_chosen, means, scales, Q_feat, pc._anchor_feat.mean())

            # if step<=15_000:
            #     # probs = torch.stack([prob, prob_adj], dim=-1)
            #     # probs = torch.softmax(probs, dim=-1)
            #     # bit_feat = pc.EG_mix_prob_2.forward(feat_chosen,
            #     #                                     mean, mean_adj,
            #     #                                     scale, scale_adj,
            #     #                                     probs[..., 0], probs[..., 1],
            #     #                                     Q=Q_feat, x_mean=pc._anchor_feat.mean())

            #     # sp_ctx = torch.zeros((feat_chosen.shape[0], 50*2), device='cuda', dtype=torch.float32)
            #     # mean, scale = pc.get_param_mlp(feat_hyper, ch_ctx, sp_ctx)
            #     # mean, scale = pc.get_param_mlp(feat_hyper, ch_ctx)
            #     # bit_feat = pc.entropy_gaussian(feat_chosen, mean, scale, Q_feat, pc._anchor_feat.mean())

            #     knn_indices = pc.knn_indices[choose_idx]
            #     sp_feat = pc._anchor_feat[knn_indices]
            #     sp_ctx = pc.get_spatial_mlp.forward(sp_feat)
            #     means, scales = pc.get_param_mlp(feat_hyper, ch_ctx, sp_ctx)
            #     bit_feat = pc.entropy_gaussian(feat_chosen, means, scales, Q_feat, pc._anchor_feat.mean())

            # else:


            #     # sp_anchor_sign = pc.spatial_anchor_sign[choose_idx]
            #     # # coords = pc.spatial_coords[visible_mask][choose_idx] consider relative coords
            #     # nac_choose_idx = choose_idx[~pc.spatial_anchor_sign]
            #     # knn_indices = pc.knn_indices[nac_choose_idx]
            #     # nac_sp_feat = pc._anchor_feat[knn_indices]

            #     # mean_sp, scale_sp, prob_sp = pc.get_spatial_mlp.forward(nac_sp_feat)

            #     # probs_ac = torch.stack([prob[sp_anchor_sign], prob_adj[sp_anchor_sign]], dim=-1)
            #     # probs_ac = torch.softmax(probs_ac, dim=-1)

            #     # probs_nac = torch.stack([prob[~sp_anchor_sign], prob_adj[~sp_anchor_sign], prob_sp], dim=-1)
            #     # probs_nac = torch.softmax(probs_nac, dim=-1)

            #     # bit_feat = torch.empty_like(feat_chosen)
            #     # bit_feat[sp_anchor_sign] = pc.EG_mix_prob_2.forward(feat_chosen[sp_anchor_sign],
            #     #                                     mean[sp_anchor_sign], mean_adj[sp_anchor_sign],
            #     #                                     scale[sp_anchor_sign], scale_adj[sp_anchor_sign],
            #     #                                     probs_ac[..., 0], probs_ac[..., 1],
            #     #                                     Q=Q_feat[sp_anchor_sign], x_mean=pc._anchor_feat.mean())
                
            #     # bit_feat[~sp_anchor_sign] = pc.EG_mix_prob_3.forward(feat_chosen[~sp_anchor_sign],
            #     #                                     mean[~sp_anchor_sign], mean_adj[~sp_anchor_sign], mean_sp,
            #     #                                     scale[~sp_anchor_sign], scale_adj[~sp_anchor_sign], scale_sp,
            #     #                                     probs_nac[..., 0], probs_nac[..., 1], probs_nac[..., 2],
            #     #                                     Q=Q_feat[~sp_anchor_sign], x_mean=pc._anchor_feat.mean())


            #     # knn_indices = pc.knn_indices[choose_idx]
            #     # sp_feat = pc._anchor_feat[knn_indices]
            #     # mean_sp, scale_sp, prob_sp = pc.get_spatial_mlp.forward(sp_feat, torch.cat([mean, scale, prob], dim=-1).contiguous())
            #     # probs = torch.stack([prob, prob_adj, prob_sp], dim=-1)
            #     # probs = torch.softmax(probs, dim=-1)
            #     # bit_feat = pc.EG_mix_prob_3.forward(feat_chosen,
            #     #                                     mean, mean_adj, mean_sp,
            #     #                                     scale, scale_adj, scale_sp,
            #     #                                     probs[..., 0], probs[..., 1], probs[..., 2],
            #     #                                     Q=Q_feat, x_mean=pc._anchor_feat.mean())


            #     # sp_anchor_sign = pc.spatial_anchor_sign[choose_idx]  
            #     # nac_choose_idx = choose_idx[~pc.spatial_anchor_sign]
            #     # knn_indices = pc.knn_indices[nac_choose_idx]
            #     # nac_sp_feat = pc._anchor_feat[knn_indices]
            #     # ac_coords = anchor_chosen[sp_anchor_sign]
            #     # ac_sp_ctx_side = pc.calc_interp_feat(ac_coords, ac_sp=True)
            #     # sp_ctx = torch.zeros((feat_chosen.shape[0], 50), device='cuda', dtype=torch.float32)
            #     # sp_ctx[sp_anchor_sign] = pc.get_ac_sp_mlp(ac_sp_ctx_side)
            #     # sp_ctx[~sp_anchor_sign] = pc.get_spatial_mlp.forward(nac_sp_feat)

            #     # knn_indices = pc.knn_indices[choose_idx]
            #     # sp_feat = pc._anchor_feat[knn_indices]
            #     # sp_ctx = torch.zeros((feat_chosen.shape[0], 50), device='cuda', dtype=torch.float32)
            #     # sp_ctx[~sp_anchor_sign] = pc.get_spatial_mlp.forward(sp_feat[~sp_anchor_sign])

            #     # ac_sp_ctx_side = pc.get_deform_mlp.forward(sp_feat[sp_anchor_sign].reshape(-1, 50))
            #     # sp_ctx[sp_anchor_sign] = pc.get_spatial_mlp.forward(ac_sp_ctx_side.reshape(-1, 3, 50))

            #     # nac_sp_ctx_side = pc.get_deform_mlp.forward(sp_feat[~sp_anchor_sign].reshape(-1, 50))
            #     # sp_ctx[~sp_anchor_sign] = pc.get_spatial_mlp.forward(nac_sp_ctx_side.reshape(-1, 3, 50))

            #     # mean, scale = pc.get_param_mlp(feat_hyper, ch_ctx, sp_ctx)
            #     # mean, scale = pc.get_param_mlp(feat_hyper, ch_ctx)
            #     # bit_feat = pc.entropy_gaussian(feat_chosen, mean, scale, Q_feat, pc._anchor_feat.mean())
                
            #     knn_indices = pc.knn_indices[choose_idx]
            #     base_indices = pc.base_mask[choose_idx]
            #     sp_feat = pc._anchor_feat[pc.refer_mask][knn_indices]
            #     # sp_feat = pc._anchor_feat[knn_indices]
            #     sp_ctx = pc.get_spatial_mlp.forward(sp_feat)
            #     sp_ctx[base_indices, ...] = 0
            #     means, scales = pc.get_param_mlp(feat_hyper, ch_ctx, sp_ctx)
            #     bit_feat = pc.entropy_gaussian(feat_chosen, means, scales, Q_feat, pc._anchor_feat.mean())

            # knn_indices = pc.knn_indices[choose_idx]
            # base_indices = pc.base_mask[choose_idx]
            # sp_feat = pc._anchor_feat[knn_indices]
            # sp_ctx = pc.get_spatial_mlp.forward(sp_feat)
            # base_pts = anchor_chosen[base_indices]
            # sp_ctx[base_indices, ...] = pc.mlp_base_sp(base_pts)
            # means, scales = pc.get_param_mlp(feat_hyper, ch_ctx, sp_ctx)
            # bit_feat = pc.entropy_gaussian(feat_chosen, means, scales, Q_feat, pc._anchor_feat.mean())


            # binary_feat_mask = torch.ones_like(bit_feat)
            # binary_feat_mask[..., pc.base_feat_dim:] = binary_grid_masks_chosen.repeat(1, 1, pc.bias_dim).view(-1, pc.bias_dim*pc.n_offsets)

            binary_grid_masks_chosen = binary_grid_masks_chosen.repeat(1, 1, 3).view(-1, 3*pc.n_offsets)

            bit_feat = bit_feat * mask_anchor_chosen
            bit_scaling = pc.entropy_gaussian.forward(grid_scaling_chosen, mean_scaling, scale_scaling, Q_scaling, pc.get_scaling.mean())
            bit_scaling = bit_scaling * mask_anchor_chosen
            bit_offsets = pc.entropy_gaussian.forward(grid_offsets_chosen, mean_offsets, scale_offsets, Q_offsets.view(-1, 3*pc.n_offsets), pc._offset.mean())
            bit_offsets = bit_offsets * mask_anchor_chosen * binary_grid_masks_chosen

            # bit_per_knn = torch.sum(bit_knn) / bit_knn.numel()
            bit_per_feat_param = torch.sum(bit_feat) / bit_feat.numel()
            bit_per_scaling_param = torch.sum(bit_scaling) / bit_scaling.numel()
            bit_per_offsets_param = torch.sum(bit_offsets) / bit_offsets.numel()
            bit_per_param = (torch.sum(bit_feat) + torch.sum(bit_scaling) + torch.sum(bit_offsets)) / \
                            (bit_feat.numel() + bit_scaling.numel() + bit_offsets.numel())

            # show different level feat bit
            # with torch.no_grad():
            #     for i, level_mask in enumerate(mask_list):
            #         level_mask_choose = level_mask[choose_idx]
            #         bit_feat_level = bit_feat[level_mask_choose]
            #         bit_per_feat_param_level = torch.sum(bit_feat_level) / bit_feat_level.numel()
            #         bit_feat_levels.append(bit_per_feat_param_level)
            #         if step%200 ==0:
            #             print(f"level_{i}_num:", bit_feat_level.shape[0])

            # if step>15_000:
            #     mask_anchor_chosen = mask_anchor_chosen[:, 0]
            #     sp_anchor_sign = pc.spatial_anchor_sign[choose_idx]
            #     valid_nac_sign = torch.logical_and(mask_anchor_chosen, ~sp_anchor_sign)
            #     bit_ac_feat = bit_feat[sp_anchor_sign]
            #     bit_nac_feat = bit_feat[valid_nac_sign]

            #     bit_per_ac_feat = torch.sum(bit_ac_feat) / bit_ac_feat.numel()
            #     bit_per_nac_feat = torch.sum(bit_nac_feat) / bit_nac_feat.numel()

    # elif not pc.decoded_version:
    #     torch.cuda.synchronize(); t1 = time.time()
    #     feat_context = pc.calc_interp_feat(anchor)
    #     mean, scale, prob, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
    #         torch.split(pc.get_grid_mlp(feat_context), split_size_or_sections=[pc.feat_dim, pc.feat_dim, pc.feat_dim, 6, 6, 3*pc.n_offsets, 3*pc.n_offsets, 1, 1, 1], dim=-1)

    #     Q_feat_adj = Q_feat_adj.contiguous().repeat(1, mean.shape[-1])
    #     Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1])
    #     Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1])
    #     Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
    #     Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
    #     Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj)).view(-1, pc.n_offsets, 3)  # [N_visible_anchor, 10, 3]
    #     feat = (STE_multistep.apply(feat, Q_feat, pc._anchor_feat.mean())).detach()
    #     grid_scaling = (STE_multistep.apply(grid_scaling, Q_scaling, pc.get_scaling.mean())).detach()
    #     grid_offsets = (STE_multistep.apply(grid_offsets, Q_offsets, pc._offset.mean())).detach()
    #     torch.cuda.synchronize(); time_sub = time.time() - t1

    else:
        pass

    ob_view = anchor - viewpoint_camera.camera_center
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)  # [3+1]

        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1)  # [N_visible_anchor, 1, 3]

        feat = feat.unsqueeze(dim=-1)  # feat: [N_visible_anchor, 32]
        feat = \
            feat[:, ::4, :1].repeat([1, 4, 1])*bank_weight[:, :, :1] + \
            feat[:, ::2, :1].repeat([1, 2, 1])*bank_weight[:, :, 1:2] + \
            feat[:, ::1, :1]*bank_weight[:, :, 2:]
        feat = feat.squeeze(dim=-1)  # [N_visible_anchor, 32]

    cat_local_view = torch.cat([ob_view, ob_dist, feat], dim=1)  # [N_visible_anchor, 32+3+1]
    
    # cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)  # [N_visible_anchor, 32+3+1]

    neural_opacity = pc.get_opacity_mlp(cat_local_view)  # [N_visible_anchor, K]
    neural_opacity = neural_opacity.reshape([-1, 1])  # [N_visible_anchor*K, 1]
    mask = (neural_opacity > 0.0)
    mask = mask.view(-1)  # [N_visible_anchor*K]

    # select opacity
    opacity = neural_opacity[mask]  # [N_opacity_pos_gaussian, 1]

    # get offset's color
    color = pc.get_color_mlp(cat_local_view)  # [N_visible_anchor, K*3]
    color = color.reshape([anchor.shape[0] * pc.n_offsets, 3])  # [N_visible_anchor*K, 3]

    # get offset's cov
    scale_rot = pc.get_cov_mlp(cat_local_view)  # [N_visible_anchor, K*7]
    scale_rot = scale_rot.reshape([anchor.shape[0] * pc.n_offsets, 7])  # [N_visible_anchor*K, 7]

    offsets = grid_offsets.view([-1, 3])  # [N_visible_anchor*K, 3]

    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)  # [N_visible_anchor, 6+3]
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)  # [N_visible_anchor*K, 6+3]
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets],
                                 dim=-1)  # [N_visible_anchor*K, (6+3)+3+7+3]
    masked = concatenated_all[mask]  # [N_opacity_pos_gaussian, (6+3)+3+7+3]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)

    # post-process cov
    scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])
    rot = pc.rotation_activation(scale_rot[:, 3:7])  # [N_opacity_pos_gaussian, 4]

    offsets = offsets * scaling_repeat[:, :3]  # [N_opacity_pos_gaussian, 3]
    xyz = repeat_anchor + offsets  # [N_opacity_pos_gaussian, 3]

    binary_grid_masks_pergaussian = binary_grid_masks.view(-1, 1)
    if is_training:
        opacity = opacity * binary_grid_masks_pergaussian[mask]
        scaling = scaling * binary_grid_masks_pergaussian[mask]
    else:
        the_mask = (binary_grid_masks_pergaussian[mask]).to(torch.bool)
        the_mask = the_mask[:, 0]
        xyz = xyz[the_mask]
        color = color[the_mask]
        opacity = opacity[the_mask]
        scaling = scaling[the_mask]
        rot = rot[the_mask]

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask, bit_per_param, bit_per_feat_param, bit_per_scaling_param, bit_per_offsets_param, bit_feat_levels, bit_per_knn, dis_loss
    else:
        return xyz, color, opacity, scaling, rot, time_sub


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, step=0):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training

    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask, bit_per_param, bit_per_feat_param, bit_per_scaling_param, bit_per_offsets_param, bit_feat_levels, bit_per_knn, dis_loss = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, step=step)
    else:
        xyz, color, opacity, scaling, rot, time_sub = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, step=step)

    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "bit_per_param": bit_per_param,
                "bit_per_feat_param": bit_per_feat_param,
                "bit_per_scaling_param": bit_per_scaling_param,
                "bit_per_offsets_param": bit_per_offsets_param,
                "bit_feat_levels": bit_feat_levels,
                "bit_per_knn": bit_per_knn,
                "dis_loss": dis_loss
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "time_sub": time_sub,
                }


def prefilter_voxel(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
                    override_color=None):
    """
    Render the scene. 

    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True,
                                          device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:  # False
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:  # into here
        scales = pc.get_scaling  # requires_grad = True
        rotations = pc.get_rotation  # requires_grad = True

    radii_pure = rasterizer.visible_filter(
        means3D=means3D,
        scales=scales[:, :3],
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,  # None
    )

    return radii_pure > 0

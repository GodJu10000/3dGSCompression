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
from utils.encodings import STE_binary, STE_multistep


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


    # bit_per_ac_feat = None
    # bit_per_nac_feat = None

    bit_per_param = None
    bit_per_feat_param = None
    bit_per_scaling_param = None
    bit_per_offsets_param = None
    bit_feat_levels = []
    Q_feat = 1
    Q_scaling = 0.001
    Q_offsets = 0.2

    if is_training:
        if step > 3000 and step <= 10000:
            # quantization
            feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat
            grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling
            grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets

        if step == 10000:
            pc.update_anchor_bound()
            # pc.encoding_xyz_config()
            pc.sp_config()
            pc.ch_config()
            pc.param_config()
            # pc.VM_config()
            pc.triplane_config()

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
            Q_feat_adj = pc.mlp_feat_Q(hyper_prior)

            Q_feat_adj = Q_feat_adj.contiguous().repeat(1, 50)
            Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1])
            Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1])

            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj)).view(-1, pc.n_offsets, 3)
            feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat
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
            choose_idx = torch.rand_like(pc.get_anchor[:, 0]) <= 0.05
            choose_idx =  choose_idx & pc.get_mask_anchor.to(torch.bool)[:, 0]

            anchor_chosen = pc.get_anchor[choose_idx]
            feat_chosen = pc._anchor_feat[choose_idx]
            grid_offsets_chosen = pc._offset[choose_idx]
            grid_scaling_chosen = pc.get_scaling[choose_idx]
            binary_grid_masks_chosen = pc.get_mask[choose_idx]  # [N_vis, 10, 1]
            mask_anchor_chosen = pc.get_mask_anchor[choose_idx]  # [N_vis, 1]

            # feat_context_orig = pc.calc_interp_feat(anchor_chosen)
            # feat_context = pc.get_grid_mlp(feat_context_orig)
            hyper_prior = pc.query_triplane(anchor_chosen)
            # mean, scale, prob, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
            #     torch.split(feat_context, split_size_or_sections=[pc.feat_dim, pc.feat_dim, pc.feat_dim, 6, 6, 3*pc.n_offsets, 3*pc.n_offsets, 1, 1, 1], dim=-1)

            # plane_context = pc.query_triplane(anchor_chosen)
            # feat_hyper, mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = \
            #     torch.split(feat_context, split_size_or_sections=[pc.feat_dim*2, 6, 6, 3*pc.n_offsets, 3*pc.n_offsets, 1, 1, 1], dim=-1)

            # feat_hyper, Q_feat_adj = torch.split(feat_context, split_size_or_sections=[pc.feat_dim*2, 1], dim=-1)
            # mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_scaling_adj, Q_offsets_adj = \
            #     torch.split(plane_context, split_size_or_sections=[6, 6, 3*pc.n_offsets, 3*pc.n_offsets, 1, 1], dim=-1)        

            mean_scaling, scale_scaling, mean_offsets, scale_offsets, Q_scaling_adj, Q_offsets_adj = \
                torch.split(pc.mlp_VM(hyper_prior), split_size_or_sections=[6, 6, 3*pc.n_offsets, 3*pc.n_offsets, 1, 1, 1], dim=-1)
            Q_feat_adj = pc.mlp_feat_Q(hyper_prior)

            Q_feat = 1
            Q_scaling = 0.001
            Q_offsets = 0.2
            Q_feat_adj = Q_feat_adj.contiguous().repeat(1, 50)
            Q_scaling_adj = Q_scaling_adj.contiguous().repeat(1, mean_scaling.shape[-1])
            Q_offsets_adj = Q_offsets_adj.contiguous().repeat(1, mean_offsets.shape[-1])
            Q_feat = Q_feat * (1 + torch.tanh(Q_feat_adj))
            Q_scaling = Q_scaling * (1 + torch.tanh(Q_scaling_adj))
            Q_offsets = Q_offsets * (1 + torch.tanh(Q_offsets_adj)).view(-1, pc.n_offsets, 3)
            feat_chosen = feat_chosen + torch.empty_like(feat_chosen).uniform_(-0.5, 0.5) * Q_feat

            # mean_adj, scale_adj, prob_adj = pc.get_deform_mlp.forward(feat_chosen, torch.cat([mean, scale, prob], dim=-1).contiguous())
            # mean_adj, scale_adj, prob_adj = pc.get_deform_mlp.forward(feat_chosen)
            # ch_ctx = pc.get_deform_mlp.forward(feat_chosen)

            # mean, scale = pc.get_param_mlp(hyper_prior, ch_ctx)
            # bit_feat = pc.entropy_gaussian(feat_chosen, mean, scale, Q_feat, pc._anchor_feat.mean())

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



            mask_list = [getattr(pc, f"level_{i}_mask") for i in range(pc.level_num)]
            knn_indices = pc.knn_indices[choose_idx]
            sp_feat = pc._anchor_feat[pc.refer_mask][pc.morton_indices][knn_indices]  # (N, knn, feat.shape[-1])
            sp_anchor = pc._anchor_feat[pc.refer_mask][pc.morton_indices][knn_indices].view(-1, 3)  # (N*knn, anchor.shape[-1])
            Q_feat_sp_adj = pc.mlp_feat_Q(pc.query_triplane(sp_anchor)).view(-1, 2, 1)  #(N, knn, 1)
            Q_feat_sp = 1
            Q_feat_sp = Q_feat_sp * (1 + torch.tanh(Q_feat_sp_adj))
            sp_feat = sp_feat + torch.empty_like(sp_feat).uniform_(-0.5, 0.5) * Q_feat_sp

            sp_ctx = pc.get_spatial_mlp.forward(sp_feat, mask_list, choose_idx)
            # sp_ctx = pc.get_spatial_mlp.forward(sp_feat)
            ch_ctx = pc.get_deform_mlp.forward(feat_chosen, mask_list, choose_idx)
            # hyper_prior = pc.calc_interp_feat(anchor_chosen)
            means, scales = pc.get_param_mlp(hyper_prior, ch_ctx, sp_ctx, mask_list, choose_idx)
            # means, scales = pc.get_param_mlp(hyper_prior, ch_ctx, sp_ctx)
            bit_feat = pc.entropy_gaussian(feat_chosen, means, scales, Q_feat, pc._anchor_feat.mean())


            with torch.no_grad():
                for level_mask in mask_list:
                    level_mask_choose = level_mask[choose_idx]
                    bit_feat_level = bit_feat[level_mask_choose]
                    bit_per_feat_param_level = torch.sum(bit_feat_level) / bit_feat_level.numel()
                    bit_feat_levels.append(bit_per_feat_param_level)

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


            grid_scaling_chosen = grid_scaling_chosen + torch.empty_like(grid_scaling_chosen).uniform_(-0.5, 0.5) * Q_scaling
            grid_offsets_chosen = grid_offsets_chosen + torch.empty_like(grid_offsets_chosen).uniform_(-0.5, 0.5) * Q_offsets
            grid_offsets_chosen = grid_offsets_chosen.view(-1, 3 * pc.n_offsets)

            binary_grid_masks_chosen = binary_grid_masks_chosen.repeat(1, 1, 3).view(-1, 3*pc.n_offsets)

            bit_feat = bit_feat * mask_anchor_chosen
            bit_scaling = pc.entropy_gaussian.forward(grid_scaling_chosen, mean_scaling, scale_scaling, Q_scaling, pc.get_scaling.mean())
            bit_scaling = bit_scaling * mask_anchor_chosen
            bit_offsets = pc.entropy_gaussian.forward(grid_offsets_chosen, mean_offsets, scale_offsets, Q_offsets.view(-1, 3*pc.n_offsets), pc._offset.mean())
            bit_offsets = bit_offsets * mask_anchor_chosen * binary_grid_masks_chosen

            bit_per_feat_param = torch.sum(bit_feat) / bit_feat.numel()
            bit_per_scaling_param = torch.sum(bit_scaling) / bit_scaling.numel()
            bit_per_offsets_param = torch.sum(bit_offsets) / bit_offsets.numel()
            bit_per_param = (torch.sum(bit_feat) + torch.sum(bit_scaling) + torch.sum(bit_offsets)) / \
                            (bit_feat.numel() + bit_scaling.numel() + bit_offsets.numel())
             
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

    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)  # [N_visible_anchor, 32+3+1]

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
        return xyz, color, opacity, scaling, rot, neural_opacity, mask, bit_per_param, bit_per_feat_param, bit_per_scaling_param, bit_per_offsets_param, bit_feat_levels
    else:
        return xyz, color, opacity, scaling, rot, time_sub


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, step=0):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training

    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask, bit_per_param, bit_per_feat_param, bit_per_scaling_param, bit_per_offsets_param, bit_feat_levels = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, step=step)
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
                "bit_feat_levels": bit_feat_levels
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

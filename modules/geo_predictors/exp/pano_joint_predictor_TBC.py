import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import numpy as np

from ..geo_predictor import GeoPredictor
from ..omnidata_predictor import OmnidataPredictor
from ..omnidata_normal_predictor import OmnidataNormalPredictor

from modules.fields.networks import VanillaMLP
import tinycudann as tcnn

from utils.geo_utils import panorama_to_pers_directions
from utils.camera_utils import *

import os
from torchvision.utils import save_image

def scale_unit(x):
    return (x - x.min()) / (x.max() - x.min())


#방향을 입력받아 distance를 출력
#실제 학습 수행
class SphereDistanceField(nn.Module): #nn.Module-학습 가능  
    def __init__(self,
                 n_levels=16,   #multi-resolution grid 레벨 수 
                 log2_hashmap_size=19, #hash table 크기
                 base_res=16, #가장 거친 해상도
                 fine_res=2048): #가장 미세한 해상도
        super().__init__()
        per_level_scale = np.exp(np.log(fine_res / base_res) / (n_levels - 1)) #base_res -> fine_res 사이를 기하급수적으로 채우도록록
        self.hash_grid = tcnn.Encoding(
            n_input_dims=3, #3d 좌표
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels, #다중해상도
                "n_features_per_level": 2, #각 레벨당 2차원 feature
                "log2_hashmap_size": log2_hashmap_size, #hash table 크기 : 2^19
                "base_resolution": base_res, #가장 거친 해상도
                "per_level_scale": per_level_scale, #레벨 간 해상도 비율
                "interpolation": "Smoothstep", #부드러운 보간(gradient 안정성)
            }
        )

        self.geo_mlp = VanillaMLP(dim_in=n_levels * 2 + 3,
                                  dim_out=1,
                                  n_neurons=64,
                                  n_hidden_layers=2,
                                  sphere_init=True,
                                  weight_norm=False)

    def forward(self, directions, requires_grad=False):
        if requires_grad:
            if not self.training:
                directions = directions.clone()  # get a copy to enable grad
            directions.requires_grad_(True)

        dir_scaled = directions * 0.49 + 0.49 #좌표 스케일링, 입력 좌표가 (0, 1)이 되도록
        selector = ((dir_scaled > 0.0) & (dir_scaled < 1.0)).all(dim=-1).to(torch.float32)  
        scene_feat = self.hash_grid(dir_scaled) #hashgrid encoding

        distance = F.softplus(self.geo_mlp(torch.cat([directions, scene_feat], -1))[..., 0] + 1.) #sdf
        # distance = torch.exp(self.geo_mlp(torch.cat([directions, scene_feat], -1))[..., 0])

        if requires_grad:
            grad = torch.autograd.grad(
                distance, directions, grad_outputs=torch.ones_like(distance),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]

            return distance, grad
        else:
            return distance


#실제 학습 수행X, pretrained model의 ckpt를 불러와서 inference만 수행
class PanoJointPredictor(GeoPredictor):
    def __init__(self):
        super().__init__()
        self.depth_predictor = OmnidataPredictor()
        self.normal_predictor = OmnidataNormalPredictor()
        # self.depth_predictor = IronPredictor()

    def grads_to_normal(self, grads):
        height, width, _ = grads.shape
        pano_dirs = img_coord_to_pano_direction(img_coord_from_hw(height, width))
        ortho_a = torch.randn([height, width, 3])
        ortho_b = torch.linalg.cross(pano_dirs, ortho_a)
        ortho_b = ortho_b / torch.linalg.norm(ortho_b, 2, -1, True)
        ortho_a = torch.linalg.cross(ortho_b, pano_dirs)
        ortho_a = ortho_a / torch.linalg.norm(ortho_a, 2, -1, True)

        val_a = (grads * ortho_a).sum(-1, True) * pano_dirs + ortho_a
        val_a = val_a / torch.linalg.norm(val_a, 2, -1, True)
        val_b = (grads * ortho_b).sum(-1, True) * pano_dirs + ortho_b
        val_b = val_b / torch.linalg.norm(val_b, 2, -1, True)

        normals = torch.cross(val_a, val_b)
        normals = normals / torch.linalg.norm(normals, 2, -1, True)
        is_inside = ((normals * pano_dirs).sum(-1, True) < 0.).float()
        normals = normals * is_inside + -normals * (1. - is_inside)
        return normals

    def __call__(self, img1, img2, ref_distance1, ref_distance2, mask, gen_res=384,
                 reg_loss_weight=1e-1, normal_loss_weight=1e-2, normal_tv_loss_weight=1e-2):
        '''
        Do not support batch operation - can only inpaint one image at a time.
        :param img: [H, W, 3]
        :param ref_distance: [H, W] or [H, W, 1]
        :param mask: [H, W] or [H, W, 1], value 1 if need to be inpainted otherwise 0
        :return: inpainted distance [H, W]
        '''

        height, width, _ = img1.shape

        device = img1.device

        img1 = img1.clone().squeeze().permute(2, 0, 1)   
        img2 = img2.clone().squeeze().permute(2, 0, 1)               
                                    # [3, H, W]
        mask = mask.clone().squeeze()[..., None].float().permute(2, 0, 1)      
                         # [1, H, W]
        ref_distance1 = ref_distance1.clone().squeeze()[..., None].float().permute(2, 0, 1)       # [1, H, W]
        ref_distance2 = ref_distance2.clone().squeeze()[..., None].float().permute(2, 0, 1)

        ref_distance_mask1 = torch.cat([ref_distance1, mask], 0)
        ref_distance_mask2 = torch.cat([ref_distance2, mask], 0)

        ref_distance_mask  = torch.stack([ref_distance_mask1, ref_distance_mask2], dim=0)

        pers_dirs, pers_ratios, to_vecs, down_vecs, right_vecs = [], [], [], [], []
        for ratio in [1.1, 1.4, 1.7]:
            cur_pers_dirs, cur_pers_ratios, cur_to_vecs, cur_down_vecs, cur_right_vecs = panorama_to_pers_directions(gen_res=gen_res, ratio=ratio, ex_rot='rand')
            pers_dirs.append(cur_pers_dirs)
            pers_ratios.append(cur_pers_ratios)
            to_vecs.append(cur_to_vecs)
            down_vecs.append(cur_down_vecs)
            right_vecs.append(cur_right_vecs)

        pers_dirs = torch.cat(pers_dirs, 0)
        pers_ratios = torch.cat(pers_ratios, 0)
        to_vecs = torch.cat(to_vecs, 0)
        down_vecs = torch.cat(down_vecs, 0)
        right_vecs = torch.cat(right_vecs, 0)

        fx = torch.linalg.norm(to_vecs, 2, -1, True) / torch.linalg.norm(right_vecs, 2, -1, True) * gen_res * .5
        fy = torch.linalg.norm(to_vecs, 2, -1, True) / torch.linalg.norm(down_vecs, 2, -1, True) * gen_res * .5
        cx = torch.ones_like(fx) * gen_res * .5
        cy = torch.ones_like(fy) * gen_res * .5

        pers_dirs = pers_dirs.to(device)
        pers_ratios = pers_ratios.to(device)
        to_vecs = to_vecs.to(device)
        down_vecs = down_vecs.to(device)
        right_vecs = right_vecs.to(device)

        rot_w2c = torch.stack([right_vecs / torch.linalg.norm(right_vecs, 2, -1, True),
                               down_vecs / torch.linalg.norm(down_vecs, 2, -1, True),
                               to_vecs / torch.linalg.norm(to_vecs, 2, -1, True)],
                              dim=1)
        rot_c2w = torch.linalg.inv(rot_w2c)

        n_pers = len(pers_dirs)
        img_coords = direction_to_img_coord(pers_dirs)
        sample_coords = img_coord_to_sample_coord(img_coords)

        pers_imgs1 = F.grid_sample(img1[None].expand(n_pers, -1, -1, -1), sample_coords, padding_mode='border') # [n_pers, 3, gen_res, gen_res]
        pers_imgs2 = F.grid_sample(img2[None].expand(n_pers, -1, -1, -1), sample_coords, padding_mode='border') # [n_pers, 3, gen_res, gen_res]
        
        # 이미지 확인
        save_dir = "./multi_view_joint(dis, norm)"
        os.makedirs(os.path.join(save_dir, "img1"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "img2"), exist_ok=True)

        for i in range(pers_imgs1.shape[0]):
            img_i1 = pers_imgs1[i].detach().cpu()  # [3, H, W]
            img_i2 = pers_imgs2[i].detach().cpu()  # [3, H, W]

            # 값 범위 보정 (중요)
            img_i1 = img_i1.clamp(0, 1)
            img_i2 = img_i2.clamp(0, 1)

            save_image(img_i1, os.path.join(save_dir, "img1", f"pers_{i:03d}.png"))
            save_image(img_i2, os.path.join(save_dir, "img2", f"pers_{i:03d}.png"))

        print(f"[DEBUG] saved perspective images to {save_dir}")
        

        #panorama -> perspective image로 변환까지 마침.
        #중심부(핵심부분)
        pred_distances_raw = []
        pred_normals_raw = []
        
        for i in range(n_pers):
            with torch.no_grad():
                intri = {
                    'fx': fx[i].item(),
                    'fy': fy[i].item(),
                    'cx': cx[i].item(),
                    'cy': cy[i].item()
                }

                pred_depth1 = self.depth_predictor.predict_depth(pers_imgs1[i: i+1], intri=intri).cuda().clip(0., None)  # [1, 1, res, res]
                pred_depth2 = self.depth_predictor.predict_depth(pers_imgs2[i: i+1], intri=intri).cuda().clip(0., None)  # [1, 1, res, res]
                
                pred_depth1 = pred_depth1 / (pred_depth1.mean() + 1e-5) #평균을 1로 맞춤. 상대적 depth만 고려. 
                pred_depth2 = pred_depth2 / (pred_depth2.mean() + 1e-5) #평균을 1로 맞춤. 상대적 depth만 고려. 
                
                pred_distances_raw.append(pred_depth1 * pers_ratios[i].permute(2, 0, 1)[None])
                pred_distances_raw.append(pred_depth2 * pers_ratios[i].permute(2, 0, 1)[None])

                pred_normals1 = self.normal_predictor.predict_normal(pers_imgs1[i: i+1])
                pred_normals2 = self.normal_predictor.predict_normal(pers_imgs2[i: i+1])
                
                pred_normals1 = pred_normals1 * 2. - 1. #[0, 1] => [-1, 1]
                pred_normals2 = pred_normals2 * 2. - 1. #[0, 1] => [-1, 1]
                
                pred_normals1 = pred_normals1 / torch.linalg.norm(pred_normals1, ord=2, dim=1, keepdim=True) #길이 1로 정규화
                pred_normals2 = pred_normals2 / torch.linalg.norm(pred_normals2, ord=2, dim=1, keepdim=True) #길이 1로 정규화

                pred_normals1 = pred_normals1.permute(0, 2, 3, 1)  # [1, 3, H, W] → [1, H, W, 3], 회전 연산을 위해
                pred_normals2 = pred_normals2.permute(0, 2, 3, 1)  

                pred_normals1 = apply_rot(pred_normals1, rot_c2w[i]) #perspective(카메라 좌표계) -> panorama(월드 좌표계)
                pred_normals2 = apply_rot(pred_normals2, rot_c2w[i]) 

                pred_normals1 = pred_normals1.permute(0, 3, 1, 2)  # [1, H, W, 3] → [1, 3, H, W], cnn 연산을 위해
                pred_normals2 = pred_normals2.permute(0, 3, 1, 2)  

                pred_normals_raw.append(pred_normals1)
                pred_normals_raw.append(pred_normals2)

        all_views = 2 * n_pers

        pred_distances_raw = torch.cat(pred_distances_raw, dim=0)  # [n_pers, 1, res, res], 하나의 tensor로 묶기(<-list)
        pred_normals_raw = torch.cat(pred_normals_raw, dim=0)      
        pers_dirs = pers_dirs.permute(0, 3, 1, 2) #[n_pers, res, res, 3] -> [n_pers, 3, res, res]

        pers_dirs_all = torch.cat([pers_dirs, pers_dirs], dim=0)

        sup_infos = torch.cat([pers_dirs_all, pred_distances_raw, pred_normals_raw], dim=1) #ray 데이터 구조체
        #[n_pers, 7, res, res]
        # sup_infos[v, :, y, x] =
        # [
        # dir_x,
        # dir_y,
        # dir_z,
        # distance,
        # normal_x,
        # normal_y,
        # normal_z
        # ]

        scale_params = torch.zeros([all_views], requires_grad=True) #depth scale 보정
        bias_params_global = torch.zeros([all_views], requires_grad=True) #bias 보정
        bias_params_local_distance  = torch.zeros([all_views, 1, gen_res, gen_res], requires_grad=True) #픽셀 단위 거리 보정
        bias_params_local_normal  = torch.zeros([all_views, 3, 128, 128], requires_grad=True) # 픽셀단위 거리 보정

        sp_dis_field = SphereDistanceField()
        # Stage 1: Optimize global parameters
        # 여기서 loss 그래프 그려보기
        # 실제로 하나의 scene으로 통합되고 있는지 확인해보기
        all_iter_steps = 1500
        lr_alpha = 1e-2
        init_lr = 1e-1
        init_lr_sp = 1e-2
        init_lr_local = 1e-1
        local_batch_size = 256

        optimizer_sp = torch.optim.Adam(sp_dis_field.parameters(), lr=init_lr_sp)
        optimizer_global = torch.optim.Adam([scale_params, bias_params_global], lr=init_lr)
        optimizer_local = torch.optim.Adam([bias_params_local_distance, bias_params_local_normal], lr=init_lr_local)

        for phase in ['global', 'hybrid']:
            for iter_step in tqdm(range(all_iter_steps)):
                progress = iter_step / all_iter_steps
                if phase == 'global':
                    progress = progress * .5
                else:
                    progress = progress * .5 + .5

                lr_ratio = (np.cos(progress * np.pi) + 1.) * (1. - lr_alpha) + lr_alpha
                for g in optimizer_global.param_groups:
                    g['lr'] = init_lr * lr_ratio
                for g in optimizer_local.param_groups:
                    g['lr'] = init_lr_local * lr_ratio
                for g in optimizer_sp.param_groups:
                    g['lr'] = init_lr_sp * lr_ratio

                idx = np.random.randint(low=0, high=all_views)
                sample_coords = torch.rand(all_views, local_batch_size, 1, 2) * 2. - 1
                cur_sup_info = F.grid_sample(sup_infos, sample_coords, padding_mode='border')        # [n_pers, 7, local_batch_size, 1]
                distance_bias = F.grid_sample(bias_params_local_distance, sample_coords, padding_mode='border')  # [n_pers, 4, local_batch_size, 1]
                distance_bias = distance_bias[:, :, :, 0].permute(0, 2, 1)
                normal_bias   = F.grid_sample(bias_params_local_normal, sample_coords, padding_mode='border')  # [n_pers, 4, local_batch_size, 1]
                normal_bias   = normal_bias[:, :, :, 0].permute(0, 2, 1)

                dirs = cur_sup_info[:, :3, :, 0].permute(0, 2, 1)                                    # [n_pers, local_batch_size, 3]
                dirs = dirs / torch.linalg.norm(dirs, 2, -1, True)

                ref_pred_distances = cur_sup_info[:, 3: 4, :, 0].permute(0, 2, 1)                         # [n_pers, local_batch_size, 1]
                ref_pred_distances = ref_pred_distances * F.softplus(scale_params[:, None, None])              # [n_pers, local_batch_size, 1]
                ref_pred_distances = ref_pred_distances + distance_bias

                ref_normals = cur_sup_info[:, 4:, :, 0].permute(0, 2, 1)
                ref_normals = ref_normals + normal_bias
                ref_normals = ref_normals / torch.linalg.norm(ref_normals, 2, -1, True)

                pred_distances, pred_grads = sp_dis_field(dirs.reshape(-1, 3), requires_grad=True)
                pred_distances = pred_distances.reshape(all_views, local_batch_size, 1)
                pred_grads = pred_grads.reshape(all_views, local_batch_size, 3)

                distance_loss = F.smooth_l1_loss(ref_pred_distances, pred_distances, beta=5e-1, reduction='mean')

                ortho_a = torch.randn([all_views, local_batch_size, 3])
                ortho_b = torch.linalg.cross(dirs, ortho_a)
                ortho_b = ortho_b / torch.linalg.norm(ortho_b, 2, -1, True)
                ortho_a = torch.linalg.cross(ortho_b, dirs)
                ortho_a = ortho_a / torch.linalg.norm(ortho_a, 2, -1, True)

                val_a = (pred_grads * ortho_a).sum(-1, True) * dirs + ortho_a
                val_a = val_a / torch.linalg.norm(val_a, 2, -1, True)
                val_b = (pred_grads * ortho_b).sum(-1, True) * dirs + ortho_b
                val_b = val_b / torch.linalg.norm(val_b, 2, -1, True)
                error_a = (val_a * ref_normals).sum(-1, True)
                error_b = (val_b * ref_normals).sum(-1, True)
                errors = torch.cat([error_a, error_b], -1)
                normal_loss = F.smooth_l1_loss(errors, torch.zeros_like(errors), beta=5e-1, reduction='mean')

                reg_loss = ((F.softplus(scale_params).mean() - 1.)**2).mean()

                if phase == 'hybrid':
                    distance_bias_local = bias_params_local_distance
                    distance_bias_tv_loss = F.smooth_l1_loss(distance_bias_local[:, :, 1:, :], distance_bias_local[:, :, :-1, :], beta=1e-2) + \
                                            F.smooth_l1_loss(distance_bias_local[:, :, :, 1:], distance_bias_local[:, :, :, :-1], beta=1e-2)
                    normal_bias_local = bias_params_local_normal
                    normal_bias_tv_loss = F.smooth_l1_loss(normal_bias_local[:, :, 1:, :], normal_bias_local[:, :, :-1, :], beta=1e-2) + \
                                          F.smooth_l1_loss(normal_bias_local[:, :, :, 1:], normal_bias_local[:, :, :, :-1], beta=1e-2)

                else:
                    distance_bias_tv_loss = 0.
                    normal_bias_tv_loss = 0.

                pano_image_coords = direction_to_img_coord(dirs.reshape(-1, 3))
                pano_sample_coords = img_coord_to_sample_coord(pano_image_coords)  # [all_batch_size, 2]
                # sampled_ref_distance_mask = F.grid_sample(ref_distance_mask[None], pano_sample_coords[None, :, None, :], padding_mode='border')  # [1, 2, batch_size, 1]
                # sampled_ref_distance = sampled_ref_distance_mask[0, 0]
                # sampled_ref_mask =     sampled_ref_distance_mask[0, 1]
                # ref_distance_loss = F.smooth_l1_loss(sampled_ref_distance.reshape(-1), pred_distances.reshape(-1), beta=1e-2, reduction='none')
                # ref_distance_loss = (ref_distance_loss * (sampled_ref_mask < .5).reshape(-1)).mean()
                
                # pano_sample_coords = img_coord_to_sample_coord(pano_image_coords)  # [all_views * local_batch_size, 2]
                # ref_distance_mask: [2, 2, H, W] — do not add a batch dim that makes 5D input (would trigger 3D grid_sample).
                # Rows of dirs alternate view-0 / view-1 per patch; sample each view's panorama separately then interleave.
                coords_grouped = pano_sample_coords.view(all_views, local_batch_size, 2).unsqueeze(2)
                coords_v0 = coords_grouped[0::2]  # [n_pers, local_batch_size, 1, 2]
                coords_v1 = coords_grouped[1::2]
                n0, n1 = coords_v0.shape[0], coords_v1.shape[0]
                inp0 = ref_distance_mask[0:1].expand(n0, -1, -1, -1)
                inp1 = ref_distance_mask[1:2].expand(n1, -1, -1, -1)
                s0 = F.grid_sample(inp0, coords_v0, padding_mode='border')  # [n_pers, 2, lbs, 1]
                s1 = F.grid_sample(inp1, coords_v1, padding_mode='border')
                s0 = s0.squeeze(-1)  # [n_pers, 2, lbs]
                s1 = s1.squeeze(-1)
                combined_d = torch.empty(all_views, local_batch_size, device=s0.device, dtype=s0.dtype)
                combined_m = torch.empty(all_views, local_batch_size, device=s0.device, dtype=s0.dtype)
                combined_d[0::2] = s0[:, 0]
                combined_d[1::2] = s1[:, 0]
                combined_m[0::2] = s0[:, 1]
                combined_m[1::2] = s1[:, 1]
                ref_distance_loss = F.smooth_l1_loss(combined_d.reshape(-1), pred_distances.reshape(-1), beta=1e-2, reduction='none')
                ref_distance_loss = (ref_distance_loss * (combined_m.reshape(-1) < 0.5)).mean()

                #loss 출력 후 그래프 그려보기
                loss = ref_distance_loss * 20. * progress + \
                       distance_loss + reg_loss * reg_loss_weight +\
                       normal_loss * normal_loss_weight +\
                       distance_bias_tv_loss * 1. +\
                       normal_bias_tv_loss * normal_tv_loss_weight

                optimizer_global.zero_grad()
                optimizer_sp.zero_grad()
                if phase == 'hybrid':
                    optimizer_local.zero_grad()

                loss.backward()
                optimizer_global.step()
                optimizer_sp.step()
                if phase == 'hybrid':
                    optimizer_local.step()

        # Get new distance map and normal map
        pano_dirs = img_coord_to_pano_direction(img_coord_from_hw(height, width))
        new_distances, new_grads = sp_dis_field(pano_dirs.reshape(-1, 3), requires_grad=True)
        new_distances = new_distances.detach().reshape(height, width, 1)
        new_normals = self.grads_to_normal(new_grads.detach().reshape(height, width, 3))

        return new_distances, new_normals




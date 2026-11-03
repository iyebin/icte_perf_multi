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
        
class FusionScene(nn.Module):
    def __init__(self, n_levels=16,   #multi-resolution grid 레벨 수 
                 log2_hashmap_size=19, #hash table 크기
                 base_res=16, #가장 거친 해상도
                 fine_res=2048):
        super().__init__()

        per_level_scale = np.exp(np.log(fine_res / base_res) / (n_levels - 1)) #base_res -> fine_res 사이를 기하급수적으로 채우도록록
        self.hash_grid = tcnn.Encoding(
            n_input_dims=2, #3d 좌표
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

        self.geo_mlp = VanillaMLP(dim_in=n_levels * 2 + 2,
                                  dim_out=1,
                                  n_neurons=64,
                                  n_hidden_layers=2,
                                  sphere_init=True,
                                  weight_norm=False)
        
    def forward(self, origins, distances, requires_grad=False):
        if requires_grad:
            if not self.training:
                origins = origins.clone()  # get a copy to enable grad
                distances = distances.clone()
            distances.requires_grad_(True)
        #distances scaled 확인해보기 
        dis_scaled = distances * 0.49 + 0.49 #좌표 스케일링, 입력 좌표가 (0, 1)이 되도록
        selector = ((dis_scaled > 0.0) & (dis_scaled < 1.0)).all(dim=-1).to(torch.float32)  
        scene_feat = self.hash_grid(dis_scaled) #hashgrid encoding

        distance = F.softplus(self.geo_mlp(torch.cat([distances, scene_feat], -1))[..., 0] + 1.) #sdf
        # distance = torch.exp(self.geo_mlp(torch.cat([directions, scene_feat], -1))[..., 0])

        if requires_grad:
            grad = torch.autograd.grad(
                distance, distances, grad_outputs=torch.ones_like(distance),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]

            return distance, grad
        else:
            return distance


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

    def __call__(self, img1, img2, ref_distance, mask, pose1=None, pose2=None, gen_res=384,
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
        img1 = img1.clone().squeeze().permute(2, 0, 1)                                            # [3, H, W]
        img2 = img2.clone().squeeze().permute(2, 0, 1)
        mask = mask.clone().squeeze()[..., None].float().permute(2, 0, 1)                       # [1, H, W]
        ref_distance = ref_distance.clone().squeeze()[..., None].float().permute(2, 0, 1)       # [1, H, W]
        ref_distance_mask = torch.cat([ref_distance, mask], 0)

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

        #변환된 perspective images
        pers_imgs1 = F.grid_sample(img1[None].expand(n_pers, -1, -1, -1), sample_coords, padding_mode='border') # [n_pers, 3, gen_res, gen_res]
        pers_imgs2 = F.grid_sample(img2[None].expand(n_pers, -1, -1, -1), sample_coords, padding_mode='border') # [n_pers, 3, gen_res, gen_res]

        # 이미지 확인
        save_dir = "./debug_pers_joint_kitchen"
        os.makedirs(save_dir, exist_ok=True)

        pers_imags1_path = os.path.joint(save_dir, "img1")
        pers_imags2_path = os.path.joint(save_dir, "img2")

        for i in range(pers_imgs1.shape[0]):
            img_i1 = pers_imgs1[i].detach().cpu()  # [3, H, W]
            img_i2 = pers_imgs2[i].detach().cpu()  # [3, H, W]

            # 값 범위 보정 (중요)
            img_i1 = img_i1.clamp(0, 1)
            img_i2 = img_i2.clamp(0, 1)

            save_image(img_i1, os.path.join(pers_imags1_path, f"pers_{i:03d}.png"))
            save_image(img_i2, os.path.join(pers_imags2_path, f"pers_{i:03d}.png"))

        print(f"[DEBUG] saved {pers_imgs1.shape[0] * 2} perspective images to {save_dir}")
        

        #panorama -> perspective image로 변환까지 마침.
        #중심부(핵심부분)
        pred_distances_raw1 = []
        pred_distances_raw2 = []
        pred_normals_raw1 = []
        pred_normals_raw2 = []

        for i in range(n_pers):
            with torch.no_grad():
                intri = {
                    'fx': fx[i].item(),
                    'fy': fy[i].item(),
                    'cx': cx[i].item(),
                    'cy': cy[i].item()
                }
                
                #image1
                pred_depth_1 = self.depth_predictor.predict_depth(pers_imgs1[i: i+1], intri=intri).cuda().clip(0., None)  # [1, 1, res, res]
                pred_depth_1 = pred_depth_1 / (pred_depth_1.mean() + 1e-5) #평균을 1로 맞춤. 상대적 depth만 고려.
                pred_distances_raw1.append(pred_depth_1 * pers_ratios[i].permute(2, 0, 1)[None])

                pred_normals_1 = self.normal_predictor.predict_normal(pers_imgs1[i: i+1])
                pred_normals_1 = pred_normals_1 * 2. - 1. #[0, 1] => [-1, 1]
                pred_normals_1 = pred_normals_1 / torch.linalg.norm(pred_normals_1, ord=2, dim=1, keepdim=True) #길이 1로 정규화화

                pred_normals_1 = pred_normals_1.permute(0, 2, 3, 1)  # [1, 3, H, W] → [1, H, W, 3], 회전 연산을 위해
                pred_normals_1 = apply_rot(pred_normals_1, rot_c2w[i]) #perspective(카메라 좌표계) -> panorama(월드 좌표계계)
                pred_normals_1 = pred_normals_1.permute(0, 3, 1, 2)  # [1, H, W, 3] → [1, 3, H, W], cnn 연산을 위해해
                pred_normals_raw1.append(pred_normals_1)

                #image2
                pred_depth_2 = self.depth_predictor.predict_depth(pers_imgs2[i: i+1], intri=intri).cuda().clip(0., None)  # [1, 1, res, res]
                pred_depth_2 = pred_depth_2 / (pred_depth_2.mean() + 1e-5) #평균을 1로 맞춤. 상대적 depth만 고려.
                pred_distances_raw2.append(pred_depth_2 * pers_ratios[i].permute(2, 0, 1)[None])

                pred_normals_2 = self.normal_predictor.predict_normal(pers_imgs2[i: i+1])
                pred_normals_2 = pred_normals_2 * 2. - 1. #[0, 1] => [-1, 1]
                pred_normals_2 = pred_normals_2 / torch.linalg.norm(pred_normals_2, ord=2, dim=1, keepdim=True) #길이 1로 정규화화

                pred_normals_2 = pred_normals_2.permute(0, 2, 3, 1)  # [1, 3, H, W] → [1, H, W, 3], 회전 연산을 위해
                pred_normals_2 = apply_rot(pred_normals_2, rot_c2w[i]) #perspective(카메라 좌표계) -> panorama(월드 좌표계계)
                pred_normals_2 = pred_normals_2.permute(0, 3, 1, 2)  # [1, H, W, 3] → [1, 3, H, W], cnn 연산을 위해해
                pred_normals_raw2.append(pred_normals_2)

        pred_distances_raw1 = torch.cat(pred_distances_raw1, dim=0)  # [n_pers, 1, res, res], 하나의 tensor로 묶기(<-list)
        pred_distances_raw2 = torch.cat(pred_distances_raw2, dim=0)  # [n_pers, 1, res, res], 하나의 tensor로 묶기(<-list)
        pred_normals_raw1 = torch.cat(pred_normals_raw1, dim=0)      # [n_pers, 3, res, res], 하나의 tensor로 묶기(<-list)
        pred_normals_raw2 = torch.cat(pred_normals_raw2, dim=0)      # [n_pers, 3, res, res], 하나의 tensor로 묶기(<-list)
        pers_dirs = pers_dirs.permute(0, 3, 1, 2) #[n_pers, res, res, 3] -> [n_pers, 3, res, res]

        sup_infos_1 = torch.cat([pers_dirs, pred_distances_raw1, pred_normals_raw1], dim=1) #ray 데이터 구조체
        sup_infos_2 = torch.cat([pers_dirs, pred_distances_raw2, pred_normals_raw2], dim=1) #ray 데이터 구조체
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

        #image1
        scale_params_1 = torch.zeros([n_pers], requires_grad=True) #depth scale 보정
        bias_params_global_1 = torch.zeros([n_pers], requires_grad=True) #bias 보정
        bias_params_local_distance_1  = torch.zeros([n_pers, 1, gen_res, gen_res], requires_grad=True) #픽셀 단위 거리 보정
        bias_params_local_normal_1  = torch.zeros([n_pers, 3, 128, 128], requires_grad=True) # 픽셀단위 거리 보정

        #image2
        scale_params_2 = torch.zeros([n_pers], requires_grad=True) #depth scale 보정
        bias_params_global_2 = torch.zeros([n_pers], requires_grad=True) #bias 보정
        bias_params_local_distance_2  = torch.zeros([n_pers, 1, gen_res, gen_res], requires_grad=True) #픽셀 단위 거리 보정
        bias_params_local_normal_2  = torch.zeros([n_pers, 3, 128, 128], requires_grad=True) # 픽셀단위 거리 보정

        sp_dis_field_1 = SphereDistanceField()
        sp_dis_field_2 = SphereDistanceField()
        # Stage 1: Optimize global parameters
        all_iter_steps = 1500
        lr_alpha = 1e-2
        init_lr = 1e-1
        init_lr_sp = 1e-2
        init_lr_local = 1e-1
        local_batch_size = 256

        #image1
        optimizer_sp_1 = torch.optim.Adam(sp_dis_field_1.parameters(), lr=init_lr_sp)
        optimizer_global_1 = torch.optim.Adam([scale_params_1, bias_params_global_1], lr=init_lr)
        optimizer_local_1 = torch.optim.Adam([bias_params_local_distance_1, bias_params_local_normal_1], lr=init_lr_local)

        #image2
        optimizer_sp_2 = torch.optim.Adam(sp_dis_field_2.parameters(), lr=init_lr_sp)
        optimizer_global_2 = torch.optim.Adam([scale_params_2, bias_params_global_2], lr=init_lr)
        optimizer_local_2 = torch.optim.Adam([bias_params_local_distance_2, bias_params_local_normal_2], lr=init_lr_local)

        for phase in ['global', 'hybrid']:
            for iter_step in tqdm(range(all_iter_steps)):
                progress = iter_step / all_iter_steps
                if phase == 'global':
                    progress = progress * .5
                else:
                    progress = progress * .5 + .5

                lr_ratio = (np.cos(progress * np.pi) + 1.) * (1. - lr_alpha) + lr_alpha
                for g in optimizer_global_1.param_groups:
                    g['lr'] = init_lr * lr_ratio
                for g in optimizer_local_1.param_groups:
                    g['lr'] = init_lr_local * lr_ratio
                for g in optimizer_sp_1.param_groups:
                    g['lr'] = init_lr_sp * lr_ratio

                for g in optimizer_global_2.param_groups:
                    g['lr'] = init_lr * lr_ratio
                for g in optimizer_local_2.param_groups:
                    g['lr'] = init_lr_local * lr_ratio
                for g in optimizer_sp_2.param_groups:
                    g['lr'] = init_lr_sp * lr_ratio

                idx = np.random.randint(low=0, high=n_pers)
                sample_coords = torch.rand(n_pers, local_batch_size, 1, 2) * 2. - 1

                cur_sup_info_1 = F.grid_sample(sup_infos_1, sample_coords, padding_mode='border')        # [n_pers, 7, local_batch_size, 1]
                cur_sup_info_2 = F.grid_sample(sup_infos_2, sample_coords, padding_mode='border')        # [n_pers, 7, local_batch_size, 1]

                distance_bias_1 = F.grid_sample(bias_params_local_distance_1, sample_coords, padding_mode='border')  # [n_pers, 4, local_batch_size, 1]
                distance_bias_1 = distance_bias_1[:, :, :, 0].permute(0, 2, 1)
                normal_bias_1   = F.grid_sample(bias_params_local_normal_1, sample_coords, padding_mode='border')  # [n_pers, 4, local_batch_size, 1]
                normal_bias_1   = normal_bias_1[:, :, :, 0].permute(0, 2, 1)

                distance_bias_2 = F.grid_sample(bias_params_local_distance_2, sample_coords, padding_mode='border')  # [n_pers, 4, local_batch_size, 1]
                distance_bias_2 = distance_bias_2[:, :, :, 0].permute(0, 2, 1)
                normal_bias_2   = F.grid_sample(bias_params_local_normal_2, sample_coords, padding_mode='border')  # [n_pers, 4, local_batch_size, 1]
                normal_bias_2   = normal_bias_2[:, :, :, 0].permute(0, 2, 1)

                dirs_1 = cur_sup_info_1[:, :3, :, 0].permute(0, 2, 1)                                    # [n_pers, local_batch_size, 3]
                dirs_1 = dirs_1 / torch.linalg.norm(dirs_1, 2, -1, True)

                dirs_2 = cur_sup_info_2[:, :3, :, 0].permute(0, 2, 1)                                    # [n_pers, local_batch_size, 3]
                dirs_2 = dirs_2 / torch.linalg.norm(dirs_2, 2, -1, True)

                ref_pred_distances_1 = cur_sup_info_1[:, 3: 4, :, 0].permute(0, 2, 1)                         # [n_pers, local_batch_size, 1]
                ref_pred_distances_1 = ref_pred_distances_1 * F.softplus(scale_params_1[:, None, None])              # [n_pers, local_batch_size, 1]
                ref_pred_distances_1 = ref_pred_distances_1 + distance_bias_1

                ref_pred_distances_2 = cur_sup_info_2[:, 3: 4, :, 0].permute(0, 2, 1)                         # [n_pers, local_batch_size, 1]
                ref_pred_distances_2 = ref_pred_distances_2 * F.softplus(scale_params_2[:, None, None])              # [n_pers, local_batch_size, 1]
                ref_pred_distances_2 = ref_pred_distances_2 + distance_bias_2

                ref_normals_1 = cur_sup_info_1[:, 4:, :, 0].permute(0, 2, 1)
                ref_normals_1 = ref_normals_1 + normal_bias_1
                ref_normals_1 = ref_normals_1 / torch.linalg.norm(ref_normals_1, 2, -1, True)

                ref_normals_2 = cur_sup_info_2[:, 4:, :, 0].permute(0, 2, 1)
                ref_normals_2 = ref_normals_2 + normal_bias_2
                ref_normals_2 = ref_normals_2 / torch.linalg.norm(ref_normals_2, 2, -1, True)

                #sphere distance field
                pred_distances_1, pred_grads_1 = sp_dis_field_1(dirs_1.reshape(-1, 3), requires_grad=True)
                pred_distances_2, pred_grads_2 = sp_dis_field_2(dirs_2.reshape(-1, 3), requires_grad=True)

                pred_distances_1 = pred_distances_1.reshape(n_pers, local_batch_size, 1)
                pred_distances_2 = pred_distances_2.reshape(n_pers, local_batch_size, 1)

                pred_grads_1 = pred_grads_1.reshape(n_pers, local_batch_size, 3)
                pred_grads_2 = pred_grads_2.reshape(n_pers, local_batch_size, 3)

                distance_loss_1 = F.smooth_l1_loss(ref_pred_distances_1, pred_distances_1, beta=5e-1, reduction='mean')
                distance_loss_2 = F.smooth_l1_loss(ref_pred_distances_2, pred_distances_2, beta=5e-1, reduction='mean')

                #image 1
                ortho_a_1 = torch.randn([n_pers, local_batch_size, 3])
                ortho_b_1 = torch.linalg.cross(dirs_1, ortho_a_1)
                ortho_b_1 = ortho_b_1 / torch.linalg.norm(ortho_b_1, 2, -1, True)
                ortho_a_1 = torch.linalg.cross(ortho_b_1, dirs_1)
                ortho_a_1 = ortho_a_1 / torch.linalg.norm(ortho_a_1, 2, -1, True)

                val_a_1 = (pred_grads_1 * ortho_a_1).sum(-1, True) * dirs_1 + ortho_a_1
                val_a_1 = val_a_1 / torch.linalg.norm(val_a_1, 2, -1, True)
                val_b_1 = (pred_grads_1 * ortho_b_1).sum(-1, True) * dirs_1 + ortho_b_1
                val_b_1 = val_b_1 / torch.linalg.norm(val_b_1, 2, -1, True)
                error_a_1 = (val_a_1 * ref_normals_1).sum(-1, True)
                error_b_1 = (val_b_1 * ref_normals_1).sum(-1, True)
                errors_1 = torch.cat([error_a_1, error_b_1], -1)
                normal_loss_1 = F.smooth_l1_loss(errors_1, torch.zeros_like(errors_1), beta=5e-1, reduction='mean')

                reg_loss_1 = ((F.softplus(scale_params_1).mean() - 1.)**2).mean()

                #image 2
                ortho_a_2 = torch.randn([n_pers, local_batch_size, 3])
                ortho_b_2 = torch.linalg.cross(dirs_2, ortho_a_2)
                ortho_b_2 = ortho_b_2 / torch.linalg.norm(ortho_b_2, 2, -1, True)
                ortho_a_2 = torch.linalg.cross(ortho_b_2, dirs_2)
                ortho_a_2 = ortho_a_2 / torch.linalg.norm(ortho_a_2, 2, -1, True)

                val_a_2 = (pred_grads_2 * ortho_a_2).sum(-1, True) * dirs_2 + ortho_a_2
                val_a_2 = val_a_2 / torch.linalg.norm(val_a_2, 2, -1, True)
                val_b_2 = (pred_grads_2 * ortho_b_2).sum(-1, True) * dirs_2 + ortho_b_2
                val_b_2 = val_b_2 / torch.linalg.norm(val_b_2, 2, -1, True)
                error_a_2 = (val_a_2 * ref_normals_2).sum(-1, True)
                error_b_2 = (val_b_2 * ref_normals_2).sum(-1, True)
                errors_2 = torch.cat([error_a_2, error_b_2], -1)
                normal_loss_2 = F.smooth_l1_loss(errors_2, torch.zeros_like(errors_2), beta=5e-1, reduction='mean')

                reg_loss_2 = ((F.softplus(scale_params_2).mean() - 1.)**2).mean()

                if phase == 'hybrid':
                    distance_bias_local_1 = bias_params_local_distance_1
                    distance_bias_tv_loss_1 = F.smooth_l1_loss(distance_bias_local_1[:, :, 1:, :], distance_bias_local_1[:, :, :-1, :], beta=1e-2) + \
                                            F.smooth_l1_loss(distance_bias_local_1[:, :, :, 1:], distance_bias_local_1[:, :, :, :-1], beta=1e-2)
                    distance_bias_local_2 = bias_params_local_distance_2
                    distance_bias_tv_loss_2 = F.smooth_l1_loss(distance_bias_local_2[:, :, 1:, :], distance_bias_local_2[:, :, :-1, :], beta=1e-2) + \
                                            F.smooth_l1_loss(distance_bias_local_2[:, :, :, 1:], distance_bias_local_2[:, :, :, :-1], beta=1e-2)
                    
                    normal_bias_local_1 = bias_params_local_normal_1
                    normal_bias_tv_loss_1 = F.smooth_l1_loss(normal_bias_local_1[:, :, 1:, :], normal_bias_local_1[:, :, :-1, :], beta=1e-2) + \
                                          F.smooth_l1_loss(normal_bias_local_1[:, :, :, 1:], normal_bias_local_1[:, :, :, :-1], beta=1e-2)
                    
                    normal_bias_local_2 = bias_params_local_normal_2
                    normal_bias_tv_loss_2 = F.smooth_l1_loss(normal_bias_local_2[:, :, 1:, :], normal_bias_local_2[:, :, :-1, :], beta=1e-2) + \
                                          F.smooth_l1_loss(normal_bias_local_2[:, :, :, 1:], normal_bias_local_2[:, :, :, :-1], beta=1e-2)

                else:
                    distance_bias_tv_loss_1, distance_bias_tv_loss_2 = 0.
                    normal_bias_tv_loss_2, normal_bias_tv_loss_2 = 0.

                #img 1
                pano_image_coords_1 = direction_to_img_coord(dirs_1.reshape(-1, 3))
                pano_sample_coords_1 = img_coord_to_sample_coord(pano_image_coords_1)  # [all_batch_size, 2]
                sampled_ref_distance_mask_1 = F.grid_sample(ref_distance_mask[None], pano_sample_coords_1[None, :, None, :], padding_mode='border')  # [1, 2, batch_size, 1]
                sampled_ref_distance_1 = sampled_ref_distance_mask_1[0, 0]
                sampled_ref_mask_1 =     sampled_ref_distance_mask_1[0, 1]
                ref_distance_loss_1 = F.smooth_l1_loss(sampled_ref_distance_1.reshape(-1), pred_distances_1.reshape(-1), beta=1e-2, reduction='none')
                ref_distance_loss_1 = (ref_distance_loss_1 * (sampled_ref_mask_1 < .5).reshape(-1)).mean()

                loss_1 = ref_distance_loss_1 * 20. * progress + \
                       distance_loss_1 + reg_loss_1 * reg_loss_weight +\
                       normal_loss_1 * normal_loss_weight +\
                       distance_bias_tv_loss_1 * 1. +\
                       normal_bias_tv_loss_1 * normal_tv_loss_weight

                optimizer_global_1.zero_grad()
                optimizer_sp_1.zero_grad()
                if phase == 'hybrid':
                    optimizer_local_1.zero_grad()

                loss_1.backward()
                optimizer_global_1.step()
                optimizer_sp_1.step()
                if phase == 'hybrid':
                    optimizer_local_1.step()

                #img 2
                pano_image_coords_2 = direction_to_img_coord(dirs_2.reshape(-1, 3))
                pano_sample_coords_2 = img_coord_to_sample_coord(pano_image_coords_2)  # [all_batch_size, 2]
                sampled_ref_distance_mask_2 = F.grid_sample(ref_distance_mask[None], pano_sample_coords_2[None, :, None, :], padding_mode='border')  # [1, 2, batch_size, 1]
                sampled_ref_distance_2 = sampled_ref_distance_mask_2[0, 0]
                sampled_ref_mask_2 =     sampled_ref_distance_mask_2[0, 1]
                ref_distance_loss_2 = F.smooth_l1_loss(sampled_ref_distance_2.reshape(-1), pred_distances_2.reshape(-1), beta=1e-2, reduction='none')
                ref_distance_loss_2 = (ref_distance_loss_2 * (sampled_ref_mask_2 < .5).reshape(-1)).mean()

                loss_2 = ref_distance_loss_2 * 20. * progress + \
                       distance_loss_2 + reg_loss_2 * reg_loss_weight +\
                       normal_loss_2 * normal_loss_weight +\
                       distance_bias_tv_loss_2 * 1. +\
                       normal_bias_tv_loss_2 * normal_tv_loss_weight

                optimizer_global_2.zero_grad()
                optimizer_sp_2.zero_grad()
                if phase == 'hybrid':
                    optimizer_local_2.zero_grad()

                loss_2.backward()
                optimizer_global_2.step()
                optimizer_sp_2.step()
                if phase == 'hybrid':
                    optimizer_local_2.step()

        # Get new distance map and normal map
        pano_dirs = img_coord_to_pano_direction(img_coord_from_hw(height, width))
        new_distances_1, new_grads_1 = sp_dis_field_1(pano_dirs.reshape(-1, 3), requires_grad=True)
        new_distances_1 = new_distances_1.detach().reshape(height, width, 1)
        new_normals_1 = self.grads_to_normal(new_grads_1.detach().reshape(height, width, 3))

        new_distances_2, new_grads_2 = sp_dis_field_2(pano_dirs.reshape(-1, 3), requires_grad=True)
        new_distances_2 = new_distances_2.detach().reshape(height, width, 1)
        new_normals_2 = self.grads_to_normal(new_grads_2.detach().reshape(height, width, 3))
        
        

        #2nd MLP
        #sp_dis_field에서 정합된 distance, normal을 하나로 합치는 과정
        # n_dis = len(new_distances_1)
        # sp_dis_field_fusion = SphereDistanceField() #??
        # scale_params_fusion = torch.zeros([n_dis], requires_grad=True)
        # bias_params_global_fusion = torch.zeros([n_dis], requires_grad=True) #bias 보정
        # bias_params_local_distance_fusion =  torch.zeros([n_dis, 1, gen_res, gen_res], requires_grad=True) #픽셀 단위 거리 보정
        # bias_params_local_normal_fusion = torch.zeros([n_dis, 3, 128, 128], requires_grad=True) # 픽셀단위 거리 보정

        # optimizer_sp_fusion = torch.optim.Adam(sp_dis_field_fusion.parameters(), lr=init_lr_sp)
        # optimizer_global_fusion = torch.optim.Adam([scale_params_fusion, bias_params_global_fusion], lr=init_lr)
        # optimizer_local_fusion = torch.optim.Adam([bias_params_local_distance_fusion, bias_params_local_normal_fusion], lr=init_lr_local)
        

       


        # renew_distances, renew_grads = sp_dis_field_fusion()

        return new_distances_1, new_normals_1, new_distances_2, new_normals_2




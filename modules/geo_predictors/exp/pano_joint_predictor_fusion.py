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
import cv2

def scale_unit(x):
    return (x - x.min()) / (x.max() - x.min())

# fusion ver.
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
            n_input_dims=3,
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
        # hash grid 제거
        # self.geo_mlp = VanillaMLP(dim_in= 6,
        #                           dim_out=1,
        #                           n_neurons=64, #width=64
        #                           n_hidden_layers=2, #depth=2
        #                           sphere_init=True,
        #                           weight_norm=False)


    #기존의 direction -> origin + direction 
    # def forward(self, origins, directions, requires_grad=False):
    #     if requires_grad:
    #         if not self.training:
    #             directions = directions.clone()  # get a copy to enable grad
    #         directions.requires_grad_(True)

    #     # dir_scaled = directions * 0.49 + 0.49 #좌표 스케일링, 입력 좌표가 (0, 1)이 되도록 <- hash grid의 전처리
    #     # selector = ((dir_scaled > 0.0) & (dir_scaled < 1.0)).all(dim=-1).to(torch.float32) <- hash grid의 전처리
    #     # scene_feat = self.hash_grid(dir_scaled) #hashgrid encoding <- hash grid의 입력값

    #     # distance = F.softplus(self.geo_mlp(torch.cat([origins, directions, scene_feat], -1))[..., 0] + 1.) #sdf
    #     distance = F.softplus(self.geo_mlp(torch.cat([origins, directions], -1))[..., 0] + 1.)
    #     # distance = torch.exp(self.geo_mlp(torch.cat([directions, scene_feat], -1))[..., 0])

        
    #     if requires_grad:
    #         grad = torch.autograd.grad(
    #             distance, directions, grad_outputs=torch.ones_like(distance),
    #             create_graph=True, retain_graph=True, only_inputs=True
    #         )[0]

    #         return distance, grad
    #     else:
    #         return distance
    
    #points 기반
    def forward(self, origins, directions, t, requires_grad=False):
        #points 직접 계산
        points = origins + t *directions
        if requires_grad:
            if not self.training:
                points = points.clone()  # get a copy to enable grad
            points.requires_grad_(True)

        # dir_scaled = directions * 0.49 + 0.49 #좌표 스케일링, 입력 좌표가 (0, 1)이 되도록 <- hash grid의 전처리
        # selector = ((dir_scaled > 0.0) & (dir_scaled < 1.0)).all(dim=-1).to(torch.float32) <- hash grid의 전처리
        # scene_feat = self.hash_grid(dir_scaled) #hashgrid encoding <- hash grid의 입력값

        # distance = F.softplus(self.geo_mlp(torch.cat([origins, directions, scene_feat], -1))[..., 0] + 1.) #sdf
        # distance = F.softplus(self.geo_mlp(torch.cat([origins, directions], -1))[..., 0] + 1.)
        # distance = torch.exp(self.geo_mlp(torch.cat([directions, scene_feat], -1))[..., 0])
        distance = F.softplus(self.geo_mlp(points)[..., 0] + 1.)
        if requires_grad:
            grad = torch.autograd.grad(
                distance, points, grad_outputs=torch.ones_like(distance),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]

            return distance, grad
        else:
            return distance


class PanoJointPredictorFusion(GeoPredictor):
    def __init__(self):
        super().__init__()
        self.depth_predictor = OmnidataPredictor()
        self.normal_predictor = OmnidataNormalPredictor()
        # self.depth_predictor = IronPredictor()

    def grads_to_normal(self, grads, device):
        height, width, _ = grads.shape
        pano_dirs = img_coord_to_pano_direction(img_coord_from_hw(height, width))
        ortho_a = torch.randn([height, width, 3], device=device)
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
    
    # @staticmethod
    # def compute_consistency_loss(field_A, field_B, pose_A, pose_B, device, n_samples=512):
    #     """
    #     A의 3D 포인트를 B의 좌표계로 변환해서 B field와 비교
        
    #     pose_A, pose_B: w2c (4x4)
    #     camera center: c = -R^T @ t
    #     """
    #     R_A, t_A = pose_A[:3, :3], pose_A[:3, 3]
    #     R_B, t_B = pose_B[:3, :3], pose_B[:3, 3]
    #     cam_A = (-R_A.T @ t_A).to(device)  # world에서 카메라 A 위치
    #     cam_B = (-R_B.T @ t_B).to(device)  # world에서 카메라 B 위치

    #     # A에서 random direction 샘플
    #     dirs_A = F.normalize(torch.randn(n_samples, 3, device=device), dim=-1)

    #     # A field에서 distance 예측
    #     dist_A = field_A(dirs_A)  # [n_samples]

    #     # A의 3D 포인트 (world space)
    #     pts_world = cam_A[None] + dirs_A * dist_A[:, None]  # [n_samples, 3]

    #     # B 카메라 기준 방향과 거리로 변환
    #     pts_from_B = pts_world - cam_B[None]                         # [n_samples, 3]
    #     dist_from_B = torch.linalg.norm(pts_from_B, dim=-1)         # [n_samples]
    #     dirs_from_B = F.normalize(pts_from_B, dim=-1)               # [n_samples, 3]

    #     # B field에서 같은 방향으로 예측
    #     dist_B_pred = field_B(dirs_from_B)  # [n_samples]

    #     # 두 field가 같은 포인트를 가리켜야 함
    #     consistency_loss = F.smooth_l1_loss(dist_B_pred, dist_from_B.detach(), beta=0.1)
    #     return consistency_loss

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

        assert pose1 is not None and pose2 is not None, "Check your poses"

        img1 = img1.clone().squeeze().permute(2, 0, 1)                                            # [3, H, W]
        img2 = img2.clone().squeeze().permute(2, 0, 1)                                            # [3, H, W]

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

        pers_imgs1 = F.grid_sample(img1[None].expand(n_pers, -1, -1, -1), sample_coords, padding_mode='border') # [n_pers, 3, gen_res, gen_res]
        pers_imgs2 = F.grid_sample(img2[None].expand(n_pers, -1, -1, -1), sample_coords, padding_mode='border') # [n_pers, 3, gen_res, gen_res]

        # 이미지 확인
        save_dir = "./debug_pers_joint_kitchen"
        os.makedirs(save_dir, exist_ok=True)

        for i in range(pers_imgs1.shape[0]):
            img_i = pers_imgs1[i].detach().cpu()  # [3, H, W]
            img_i2 = pers_imgs2[i].detach().cpu()  # [3, H, W]

            # 값 범위 보정 
            img_i = img_i.clamp(0, 1)
            img_i2 = img_i2.clamp(0, 1)

            save_image(img_i, os.path.join(save_dir, f"pers_{i:03d}.png"))
            save_image(img_i2, os.path.join(save_dir, f"pers_{i:03d}_2.png"))

        print(f"[DEBUG] saved {pers_imgs1.shape[0]} perspective images to {save_dir}")
        

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
                pred_depth1 = self.depth_predictor.predict_depth(pers_imgs1[i: i+1], intri=intri).cuda().clip(0., None)  # [1, 1, res, res]
                pred_depth1 = pred_depth1 / (pred_depth1.mean() + 1e-5) #평균을 1로 맞춤. 상대적 depth만 고려.
                pred_distances_raw1.append(pred_depth1 * pers_ratios[i].permute(2, 0, 1)[None])

                pred_normals1 = self.normal_predictor.predict_normal(pers_imgs1[i: i+1])
                pred_normals1 = pred_normals1 * 2. - 1. #[0, 1] => [-1, 1]
                pred_normals1 = pred_normals1 / torch.linalg.norm(pred_normals1, ord=2, dim=1, keepdim=True) #길이 1로 정규화

                pred_normals1 = pred_normals1.permute(0, 2, 3, 1)  # [1, 3, H, W] → [1, H, W, 3], 회전 연산을 위해
                pred_normals1 = apply_rot(pred_normals1, rot_c2w[i]) #perspective(카메라 좌표계) -> panorama(월드 좌표계계)
                pred_normals1 = pred_normals1.permute(0, 3, 1, 2)  # [1, H, W, 3] → [1, 3, H, W], cnn 연산을 위해
                pred_normals_raw1.append(pred_normals1)

                #image 2
                pred_depth2 = self.depth_predictor.predict_depth(pers_imgs2[i: i+1], intri=intri).cuda().clip(0., None)  # [1, 1, res, res]
                pred_depth2 = pred_depth2 / (pred_depth2.mean() + 1e-5) #평균을 1로 맞춤. 상대적 depth만 고려.
                pred_distances_raw2.append(pred_depth2 * pers_ratios[i].permute(2, 0, 1)[None])

                pred_normals2 = self.normal_predictor.predict_normal(pers_imgs2[i: i+1])
                pred_normals2 = pred_normals2 * 2. - 1. #[0, 1] => [-1, 1]
                pred_normals2 = pred_normals2 / torch.linalg.norm(pred_normals2, ord=2, dim=1, keepdim=True) #길이 1로 정규화화            

                pred_normals2 = pred_normals2.permute(0, 2, 3, 1)  # [1, 3, H, W] → [1, H, W, 3], 회전 연산을 위해
                pred_normals2 = apply_rot(pred_normals2, rot_c2w[i]) #perspective(카메라 좌표계) -> panorama(월드 좌표계계)
                pred_normals2 = pred_normals2.permute(0, 3, 1, 2)  # [1, H, W, 3] → [1, 3, H, W], cnn 연산을 위해
                pred_normals_raw2.append(pred_normals2)

        #OmnidataPredictor 최종 결과
        # breakpoint()
        pred_distances_raw1 = torch.cat(pred_distances_raw1, dim=0)  # 60, 1, 384, 384
        pred_distances_raw2 = torch.cat(pred_distances_raw2, dim=0)  # 60, 1, 384, 384
        #OmnidataNormalPredictor 최종 결과
        pred_normals_raw1 = torch.cat(pred_normals_raw1, dim=0)      # 60, 3, 384, 384
        pred_normals_raw2 = torch.cat(pred_normals_raw2, dim=0)      # 60, 3, 384, 384
        pers_dirs = pers_dirs.permute(0, 3, 1, 2).to(device)         # 60, 3, 384, 384

        # origin 계산 (w2c → camera center in world)
        # if pose1 is not None:
        #     R_A, t_A = pose1[:3, :3], pose1[:3, 3]
        #     cam_center_A = (-R_A.T @ t_A).to(device)      # [3]
        # if pose2 is not None:
        #     R_B, t_B = pose2[:3, :3], pose2[:3, 3]
        #     cam_center_B = (-R_B.T @ t_B).to(device)      # [3]

        # assert pose1 is None or pose2 is None, "Check your pose inputs."

        # n_total = n_pers * 2 #120

        #shape 체크
        origin1 = torch.tensor([1, 1, 0], device=device)                            # [3]
        origin1 = origin1[None, :, None, None].expand( n_pers, 3, gen_res, gen_res ) # 60, 3, 384, 384
        origin2 = torch.tensor([0, 0, 0], device=device)                             # [3]
        origin2 = origin2[None, :, None, None].expand( n_pers, 3, gen_res, gen_res ) # 60, 3, 384, 384
        ray_origins = torch.cat([origin1, origin2], dim=0) #120, 3, 384, 384
        # origin1 = cam_center_A[None, :, None, None].expand(n_pers, 3, gen_res, gen_res) # 60, 3, 384, 384
        # origin2 = cam_center_B[None, :, None, None].expand(n_pers, 3, gen_res, gen_res) # 60, 3, 384, 384
        # ray_origins   = torch.cat([origin1, origin2], dim=0) # 120, 3, 384, 384
        
        ray_dirs    = torch.cat([pers_dirs, pers_dirs], dim=0)                       # 120, 3, 384, 384
        ray_distances  = torch.cat([pred_distances_raw1, pred_distances_raw2], dim=0)# 120, 1, 384, 384
        ray_normals = torch.cat([pred_normals_raw1, pred_normals_raw2], dim=0) #120, 3, 384, 384
        #To be check
        # sup_infos_a = torch.cat([pers_dirs, pred_distances_raw1, pred_normals_raw1], dim=1) 
        # sup_infos_b = torch.cat([pers_dirs, pred_distances_raw2, pred_normals_raw2], dim=1) 
        sup_infos = torch.cat([ray_origins, ray_dirs, ray_distances, ray_normals], dim=1) #120, 10, 384, 384
        
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
        num_view = n_pers * 2
        scale_params = torch.zeros([num_view], requires_grad=True, device=device) #depth scale 보정
        bias_params_global = torch.zeros([num_view], requires_grad=True, device=device) #bias 보정
        bias_params_local_distance  = torch.zeros([num_view, 1, gen_res, gen_res], requires_grad=True, device=device) #픽셀 단위 거리 보정
        bias_params_local_normal  = torch.zeros([num_view, 3, 128, 128], requires_grad=True, device=device) # 픽셀단위 거리 보정
        #image2
        # scale_params_b = torch.zeros([n_pers], requires_grad=True, device=device) #depth scale 보정
        # bias_params_global_b = torch.zeros([n_pers], requires_grad=True, device=device) #bias 보정
        # bias_params_local_distance_b  = torch.zeros([n_pers, 1, gen_res, gen_res], requires_grad=True, device=device) #픽셀 단위 거리 보정
        # bias_params_local_normal_b  = torch.zeros([n_pers, 3, 128, 128], requires_grad=True, device=device) # 픽셀단위 거리 보정

        sp_dis_field = SphereDistanceField().to(device)
        # sp_dis_field_b = SphereDistanceField().to(device)
        # Stage 1: Optimize global parameters
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

                idx = np.random.randint(low=0, high=num_view)
                sample_coords = torch.rand(num_view, local_batch_size, 1, 2, device=device) * 2. - 1

                # distance_bias = F.grid_sample(bias_params_local_distance, sample_coords, padding_mode='border')  # [n_pers, 4, local_batch_size, 1]
                # distance_bias = distance_bias[:, :, :, 0].permute(0, 2, 1)
                # normal_bias   = F.grid_sample(bias_params_local_normal, sample_coords, padding_mode='border')  # [n_pers, 4, local_batch_size, 1]
                # normal_bias   = normal_bias[:, :, :, 0].permute(0, 2, 1)

                # 각각
                #image1
                cur_sup = F.grid_sample(sup_infos, sample_coords, padding_mode='border')        # [n_pers, 7, local_batch_size, 1]
                
                distance_bias = F.grid_sample(bias_params_local_distance, sample_coords, padding_mode='border')  # [n_pers, 4, local_batch_size, 1]
                distance_bias = distance_bias[:, :, :, 0].permute(0, 2, 1)
                normal_bias   = F.grid_sample(bias_params_local_normal, sample_coords, padding_mode='border')  # [n_pers, 4, local_batch_size, 1]
                normal_bias   = normal_bias[:, :, :, 0].permute(0, 2, 1)

                origins = cur_sup[:, 0:3, :, 0].permute(0, 2, 1)
                dirs = cur_sup[:, 3:6, :, 0].permute(0, 2, 1)                                    # [n_pers, local_batch_size, 3]
                dirs = dirs / torch.linalg.norm(dirs, 2, -1, True)
                ref_pred_distances = cur_sup[:, 6:7, :, 0].permute(0, 2, 1)                         # [n_pers, local_batch_size, 1]
                ref_pred_distances = ref_pred_distances * F.softplus(scale_params[:, None, None])              # [n_pers, local_batch_size, 1]
                ref_pred_distances = ref_pred_distances + distance_bias
                ref_normals = cur_sup[:, 7:, :, 0].permute(0, 2, 1)
                ref_normals = ref_normals + normal_bias
                ref_normals = ref_normals / torch.linalg.norm(ref_normals, 2, -1, True)

                pred_distances, pred_grads = sp_dis_field(origins.reshape(-1, 3), dirs.reshape(-1, 3), requires_grad=True)

                pred_distances = pred_distances.reshape(num_view, local_batch_size, 1)
                pred_grads = pred_grads.reshape(num_view, local_batch_size, 3)
                
                distance_loss = F.smooth_l1_loss(ref_pred_distances, pred_distances, beta=5e-1, reduction='mean')

                ortho_a = torch.randn([num_view, local_batch_size, 3], device=device)
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

                #image 2
                # cur_sup_b = F.grid_sample(sup_infos_b, sample_coords, padding_mode='border')        # [n_pers, 7, local_batch_size, 1]

                # distance_bias_b = F.grid_sample(bias_params_local_distance_b, sample_coords, padding_mode='border')  # [n_pers, 4, local_batch_size, 1]
                # distance_bias_b = distance_bias_b[:, :, :, 0].permute(0, 2, 1)
                # normal_bias_b   = F.grid_sample(bias_params_local_normal_b, sample_coords, padding_mode='border')  # [n_pers, 4, local_batch_size, 1]
                # normal_bias_b   = normal_bias_b[:, :, :, 0].permute(0, 2, 1)

                # dirs_b = cur_sup_b[:, :3, :, 0].permute(0, 2, 1)                                    # [n_pers, local_batch_size, 3]
                # dirs_b = dirs_b / torch.linalg.norm(dirs_b, 2, -1, True)
                # ref_pred_distances_b = cur_sup_b[:, 3: 4, :, 0].permute(0, 2, 1)                         # [n_pers, local_batch_size, 1]
                # ref_pred_distances_b = ref_pred_distances_b * F.softplus(scale_params_b[:, None, None])              # [n_pers, local_batch_size, 1]
                # ref_pred_distances_b = ref_pred_distances_b + distance_bias_b
                # ref_normals_b = cur_sup_b[:, 4:, :, 0].permute(0, 2, 1)
                # ref_normals_b = ref_normals_b + normal_bias_b
                # ref_normals_b = ref_normals_b / torch.linalg.norm(ref_normals_b, 2, -1, True)

                # pred_distances_b, pred_grads_b = sp_dis_field_b(dirs_b.reshape(-1, 3), requires_grad=True)

                # pred_distances_b = pred_distances_b.reshape(n_pers, local_batch_size, 1)
                # pred_grads_b = pred_grads_b.reshape(n_pers, local_batch_size, 3)

                # distance_loss_b = F.smooth_l1_loss(ref_pred_distances_b, pred_distances_b, beta=5e-1, reduction='mean')

                # ortho_a_2 = torch.randn([n_pers, local_batch_size, 3], device=device)
                # ortho_b_2 = torch.linalg.cross(dirs_b, ortho_a_2)
                # ortho_b_2 = ortho_b_2 / torch.linalg.norm(ortho_b_2, 2, -1, True)
                # ortho_a_2 = torch.linalg.cross(ortho_b_2, dirs_b)
                # ortho_a_2 = ortho_a_2 / torch.linalg.norm(ortho_a_2, 2, -1, True)

                # val_a_2 = (pred_grads_b * ortho_a_2).sum(-1, True) * dirs_b + ortho_a_2
                # val_a_2 = val_a_2 / torch.linalg.norm(val_a_2, 2, -1, True)
                # val_b_2 = (pred_grads_b * ortho_b_2).sum(-1, True) * dirs_b + ortho_b_2
                # val_b_2 = val_b_2 / torch.linalg.norm(val_b_2, 2, -1, True)
                # error_a_2 = (val_a_2 * ref_normals_b).sum(-1, True)
                # error_b_2 = (val_b_2 * ref_normals_b).sum(-1, True)
                # errors_2 = torch.cat([error_a_2, error_b_2], -1)
                # normal_loss_b = F.smooth_l1_loss(errors_2, torch.zeros_like(errors_2), beta=5e-1, reduction='mean')

                # reg_loss_b = ((F.softplus(scale_params_b).mean() - 1.)**2).mean()
                

                # origins = cur_sup_info[:, :3, :, 0].permute(0,2,1)
                # # dirs = cur_sup_info[:, :3, :, 0].permute(0, 2, 1)                                    # [n_pers, local_batch_size, 3]
                # dirs = cur_sup_info[:, 3: 6, :, 0].permute(0, 2, 1) 
                

                # ref_pred_distances = cur_sup_info[:, 3: 4, :, 0].permute(0, 2, 1)                         # [n_pers, local_batch_size, 1]
                # ref_pred_distances = cur_sup_info[:, 6:7, :, 0].permute(0,2,1)
                

                # ref_normals = cur_sup_info[:, 7:, :, 0].permute(0,2,1)
                # ref_normals = cur_sup_info[:, 4:, :, 0].permute(0, 2, 1)
                # ref_normals = ref_normals + normal_bias
                # ref_normals = ref_normals / torch.linalg.norm(ref_normals, 2, -1, True)

                # pred_distances, pred_grads = sp_dis_field(origins.reshape(-1,3), dirs.reshape(-1, 3), requires_grad=True)

                pano_image_coords = direction_to_img_coord(dirs.reshape(-1, 3))
                pano_sample_coords = img_coord_to_sample_coord(pano_image_coords)  # [all_batch_size, 2]
                sampled_ref_distance_mask = F.grid_sample(ref_distance_mask[None], pano_sample_coords[None, :, None, :], padding_mode='border')  # [1, 2, batch_size, 1]
                sampled_ref_distance = sampled_ref_distance_mask[0, 0]
                sampled_ref_mask =     sampled_ref_distance_mask[0, 1]
                ref_distance_loss = F.smooth_l1_loss(sampled_ref_distance.reshape(-1), pred_distances.reshape(-1), beta=1e-2, reduction='none')
                ref_distance_loss = (ref_distance_loss * (sampled_ref_mask < .5).reshape(-1)).mean()

                # consistency_loss = self.compute_consistency_loss(
                #         sp_dis_field_a, sp_dis_field_b,
                #         pose1, pose2, device
                # )

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
        pano_dirs = img_coord_to_pano_direction(img_coord_from_hw(height, width)).to(device)
        # origin_ref = cam_center_A[None].expand(height, width, 3) # [H, W, 3]
        # 두 origin의 중간점
        mid_origin = torch.tensor([0.5, 0.5, 0.], device=device)  # ([1,1,0] + [0,0,0]) / 2
        pano_origins = mid_origin[None].expand(height * width, 3)  # [H*W, 3]

        new_distances, new_grads = sp_dis_field(pano_origins.reshape(-1, 3), pano_dirs.reshape(-1, 3), requires_grad=True)
        new_distances = new_distances.detach().reshape(height, width, 1)
        # new_distances = new_distances.detach().reshape(height, width)  # [H, W, 1] → [H, W]
        new_normals = self.grads_to_normal(new_grads.detach().reshape(height, width, 3), device=device)


        return new_distances, new_normals
    

    

# if __name__ == "__main__":
#     predictor = PanoJointPredictor()
#     img1 = "/data/intern01/PeRF_multi/example_data/BMW_near/321_331/row321.jpg"
#     img2 = "/data/intern01/PeRF_multi/example_data/BMW_near/321_331/row331.jpg"
#     # img1 = torch.from_numpy(cv2.imread(img1)).cuda()
#     # img2 = torch.from_numpy(cv2.imread(img2)).cuda()
#     img1 = torch.from_numpy(cv2.imread(img1)).float() / 255.0  # uint8 → float32
#     img2 = torch.from_numpy(cv2.imread(img2)).float() / 255.0  # uint8 → float32

#     img1 = img1.cuda()
#     img2 = img2.cuda()
#     breakpoint()
#     ref_distance = torch.ones([img1.shape[0], img1.shape[1], 1]).cuda() #2560, 5120, 1
#     mask = torch.ones([img1.shape[0], img1.shape[1]]).cuda() # 2560, 5120
#     # predictor(img1, img2, ref_distance, mask)

#     try:
#         predictor(img1, img2, ref_distance, mask)
#     except RuntimeError as e:
#         import traceback
#         traceback.print_exc()  # 정확한 라인 번호 포함한 전체 stack trace 출력





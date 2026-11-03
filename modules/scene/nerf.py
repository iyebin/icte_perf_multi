import os
import itertools
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import cv2 as cv
from kornia.morphology import erosion, dilation

from .scene import Scene
from .nerf_renderer import NeRFPropRenderer, NeRFOCCRenderer
from modules.fields.ngp_nerf import NGPNeRF
from modules.fields.ngp_nerf import NGPDensityField


from utils.camera_utils import *
from modules.dataset.sup_info import SupInfoPool

from torch_efficient_distloss import flatten_eff_distloss, eff_distloss
from nerfacc.estimators.prop_net import PropNetEstimator
from nerfacc.estimators.occ_grid import OccGridEstimator

from utils.utils import write_image, colorize_single_channel_image

class NeRFScene(Scene):
    def __init__(self,
                 base_exp_dir,
                 train_conf,
                 estimator_type,
                 renderer_conf):
        super().__init__()
        self.aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
        self.base_exp_dir = base_exp_dir
        self.writer = SummaryWriter(log_dir=pjoin(base_exp_dir, 'ts_log'))
        self.train_conf = train_conf
        self.nerf = NGPNeRF(aabb=self.aabb)
        # self.optimizer = torch.optim.Adam(self.nerf.parameters(), lr=self.train_conf.optimizer.init_lr)

        if estimator_type == 'prop':
            # proposal networks
            self.prop_networks = [
                NGPDensityField(
                    aabb=self.aabb,
                    unbounded=False,
                    n_levels=5,
                    max_resolution=128,
                ),
                NGPDensityField(
                    aabb=self.aabb,
                    unbounded=False,
                    n_levels=5,
                    max_resolution=256,
                ),
            ]
            self.prop_optimizer = torch.optim.Adam(
                itertools.chain(*[p.parameters() for p in self.prop_networks]),
                lr=self.train_conf.prop_optimizer.init_lr,
                eps=1e-15,
                betas=(0.9, 0.99),
                weight_decay=1e-6
            )
            self.estimator = PropNetEstimator(self.prop_optimizer, None).cuda()
            self.renderer = NeRFPropRenderer(**renderer_conf)
        else:
            self.estimator = OccGridEstimator(roi_aabb=self.aabb, resolution=256, levels=1).cuda()
            self.renderer = NeRFOCCRenderer(**renderer_conf)

        self.global_iter_step_geo = 0
        self.global_iter_step_app = 0

    @torch.no_grad()
    def render(self, rays: Rays, query_keys=('rgb',), sampling_requires_grad=False):
        last_train = self.nerf.training
        self.set_eval()
        rays_o, rays_d = rays.collapse()
        pre_shape = list(rays_o.shape[:-1])
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        ret = dict()
        for query_key in query_keys:
            ret[query_key] = []

        batch_size = 32768
        rays_o_batches = rays_o.split(batch_size)
        rays_d_batches = rays_d.split(batch_size)
        for rays_o_batch, rays_d_batch in zip(rays_o_batches, rays_d_batches):
            cur_render_result = self.render_once(Rays(rays_o_batch, rays_d_batch), query_keys, sampling_requires_grad=sampling_requires_grad)
            for query_key in query_keys:
                ret[query_key] += cur_render_result[query_key]

        for query_key in query_keys:
            ret[query_key] = torch.concatenate(ret[query_key], dim=0).reshape(pre_shape + [-1])

        if last_train:
            self.set_train()
        return ret

    def render_once(self, rays: Rays, query_keys=('rgb',), sampling_requires_grad=False, geo_inference=False, app_inference=False):
        rays = self.to_bounded_rays(rays)
        rays_o, rays_d, near, far = rays.collapse()
        assert len(rays_o.shape) == 2

        if isinstance(self.renderer, NeRFPropRenderer):
            render_result = self.renderer.render(self.nerf, self.prop_networks, self.estimator,
                                                 rays_o, rays_d, near, far,
                                                 sampling_requires_grad=sampling_requires_grad)
        else:
            render_result = self.renderer.render(self.nerf, self.estimator,
                                                 rays_o, rays_d, near, far,
                                                 geo_inference=geo_inference,
                                                 app_inference=app_inference)
        if (render_result is None) or (not render_result['is_valid']):
            return render_result

        ret_dict = dict()
        query_keys_ex = list(query_keys) + ['is_valid']
        for query_key in query_keys_ex:
            ret_dict[query_key] = render_result[query_key]

        return ret_dict

    def fit(self, sup_pool: SupInfoPool):
        if len(sup_pool.sup_infos) == 1:
            self.train_one_episode(sup_pool,
                                   self.train_conf.raw_phase_iter_geo,
                                   self.train_conf.raw_phase_iter_app,
                                   pixel_sup_rand_mode='by_all_pixels')
        else:
            self.train_one_episode(sup_pool,
                                   self.train_conf.raw_phase_iter_geo,
                                   self.train_conf.raw_phase_iter_app,
                                   pixel_sup_rand_mode='by_all_pixels')

    def train_one_episode(self, sup_pool: SupInfoPool, geo_res_iters, app_res_iters, pixel_sup_rand_mode):
        self.set_train()
        grad_scaler = torch.cuda.amp.GradScaler(2**7)
        pre_grid = None
        occ_res = 256

        if isinstance(self.estimator, OccGridEstimator):
            self.estimator = OccGridEstimator(roi_aabb=self.aabb, resolution=256, levels=1).cuda()
            pre_grid, occ_pts = sup_pool.gen_occ_grid(res=occ_res) 

            #한겹
            #  # --- dilation 추가 ---
            # print('occupied before:', pre_grid.sum().item())
            # g = pre_grid.float().reshape(1, 1, occ_res, occ_res, occ_res)
            # g = F.max_pool3d(g, kernel_size=3, stride=1, padding=1)
            # pre_grid = (g.reshape(-1) > 0.5)
            # print('occupied after :', pre_grid.sum().item())
            # # ---------------------

            #두겹
            # --- dilation 추가 ---
            # print('occupied before:', pre_grid.sum().item())
            # g = pre_grid.float().reshape(1, 1, occ_res, occ_res, occ_res)
            # g = F.max_pool3d(g, kernel_size=5, stride=1, padding=2)   # 3→5, padding 1→2
            # pre_grid = (g.reshape(-1) > 0.5)
            # print('occupied after :', pre_grid.sum().item())
            # ---------------------

        def occ_eval_fn(x):
            if pre_grid is None:
                density = self.nerf.query_density(x)
                return density * 5e-3
            else:
                x = (x.clip(-0.999, 0.999) * .5 + .5) * occ_res
                x = x.to(torch.int64)
                idx = x[..., 0] * occ_res * occ_res +\
                      x[..., 1] * occ_res +\
                      x[..., 2]
                return pre_grid[idx].float()

        if pre_grid is not None:
            for i in tqdm(range(256)):
                self.estimator.update_every_n_steps(
                    step=i,
                    occ_eval_fn=occ_eval_fn,
                    occ_thre=1e-2,
                    ema_decay=0.1,
                    warmup_steps=256,
                    n=1,
                )

        self.nerf.reset_geo() #  = NGPNeRF(aabb=self.aabb)
        geo_optimizer = torch.optim.Adam(self.nerf.geo_mlp.parameters(), lr=self.train_conf.geo_optimizer.init_lr)

        for iter_i in tqdm(range(geo_res_iters)):
            self.update_lr(geo_optimizer, self.train_conf.geo_optimizer, iter_i / geo_res_iters)
            if hasattr(self, 'prop_optimizer'):
                self.update_lr(self.prop_optimizer, self.train_conf.prop_optimizer, iter_i / geo_res_iters)

            self.train_one_step_geo(geo_optimizer, sup_pool, pixel_sup_rand_mode, progress=iter_i / app_res_iters, grad_scaler=grad_scaler)

        app_optimizer = torch.optim.Adam(self.nerf.app_mlp.parameters(), lr=self.train_conf.app_optimizer.init_lr)

        for iter_i in tqdm(range(app_res_iters)):
            self.update_lr(app_optimizer, self.train_conf.app_optimizer, iter_i / app_res_iters)
            self.train_one_step_app(app_optimizer, sup_pool, pixel_sup_rand_mode, progress=iter_i / app_res_iters, grad_scaler=grad_scaler)

    def train_one_step_geo(self, optimizer, sup_pool: SupInfoPool, pixel_sup_rand_mode, progress, grad_scaler=None):
        train_conf = self.train_conf
        loss = 0.
        eps = 1e-7
        optimizer.zero_grad()

        batch_size = train_conf.pixel_loss_batch_size
        rays, gt_colors, gt_depths, gt_normals = sup_pool.rand_ray_color_data(batch_size,
                                                                              rand_mode=pixel_sup_rand_mode)

        if isinstance(self.renderer, NeRFOCCRenderer):
            query_keys = ['rgb', 'distance', 'weights', 't_starts', 't_ends', 'trans', 'ray_indices', 'opacities']  # ← 'opacities' 추가
        else:
            query_keys = ['rgb', 'distance', 'weights', 't_starts', 't_ends', 'trans', 'opacities']  # ← 여기도
        render_result = self.render_once(rays, query_keys=query_keys,
                                         sampling_requires_grad=self.need_to_update_occ(self.global_iter_step_geo),
                                         app_inference=True)

        if (render_result is None) or (not render_result['is_valid']):
            self.global_iter_step_geo += 1
            return

        depth_loss_weight = train_conf.depth_loss_weight #1
        if depth_loss_weight > eps:
            pred_depths = render_result['distance']
            depth_loss = F.smooth_l1_loss(pred_depths, gt_depths, beta=1e-2, reduction='mean')
            loss = loss + depth_loss * depth_loss_weight
            self.writer.add_scalar('nerf_loss/depth_loss', depth_loss, self.global_iter_step_geo)

        # distortion loss
        distortion_loss_weight = train_conf.distortion_loss_weight #0.1
        if distortion_loss_weight > eps:
            if isinstance(self.renderer, NeRFPropRenderer):
                weights = render_result['weights']
                mid_dis = (render_result['t_ends'] + render_result['t_starts']) * .5
                sec_lens = render_result['t_ends'] - render_result['t_starts']
                dist_loss = eff_distloss(weights, mid_dis, sec_lens)
                ratio = progress
                loss = loss + dist_loss * distortion_loss_weight * ratio
            else:
                weights = render_result['weights']
                mid_dis = (render_result['t_ends'] + render_result['t_starts']) * .5
                sec_lens = render_result['t_ends'] - render_result['t_starts']
                ray_indices = render_result['ray_indices']
                dist_loss = flatten_eff_distloss(weights, mid_dis, sec_lens, ray_indices)
                # if progress < 0.0:
                #     ratio = 0.
                # else:
                #     local_progress = (progress - 0.1) / 0.9
                ratio = np.min([progress * 2., 1])
                loss = loss + dist_loss * distortion_loss_weight * ratio

            self.writer.add_scalar('nerf_loss/dist_loss', dist_loss, self.global_iter_step_geo)
        
        # ── 여기에 opacity loss 추가 ──
        opacity_loss_weight = train_conf.opacity_loss_weight
        if opacity_loss_weight > eps:
            opacities = render_result['opacities']
            opacity_loss = F.l1_loss(opacities, torch.ones_like(opacities))
            loss = loss + opacity_loss * opacity_loss_weight
            self.writer.add_scalar('nerf_loss/opacity_loss', opacity_loss, self.global_iter_step_geo)
        # ────────────────────────────

        density_loss_weight = train_conf.density_loss_weight
        if density_loss_weight > eps:
            rand_pts = (torch.rand(8192, 3) * 2. - 1.) * 0.99
            density = self.nerf.query_density(rand_pts)
            density_loss = density.mean()
            loss = loss + density_loss * density_loss_weight

            self.writer.add_scalar('nerf_loss/density_loss', density_loss, self.global_iter_step_geo)

        if grad_scaler is None:
            loss.backward()
        else:
            grad_scaler.scale(loss).backward()
        optimizer.step()

        self.writer.add_scalar('others/lr_geo', optimizer.param_groups[0]['lr'], self.global_iter_step_geo)

        self.global_iter_step_geo += 1

    def train_one_step_app(self, optimizer, sup_pool: SupInfoPool, pixel_sup_rand_mode, progress, grad_scaler=None):
        train_conf = self.train_conf
        loss = 0.
        eps = 1e-7
        optimizer.zero_grad()

        batch_size = train_conf.pixel_loss_batch_size
        rays, gt_colors, gt_depths, gt_normals = sup_pool.rand_ray_color_data(batch_size,
                                                                              rand_mode=pixel_sup_rand_mode)

        if isinstance(self.renderer, NeRFOCCRenderer):
            query_keys = ['rgb', 'distance', 'weights', 't_starts', 't_ends', 'trans', 'ray_indices', 'opacities']  # ← 'opacities' 추가
        else:
            query_keys = ['rgb', 'distance', 'weights', 't_starts', 't_ends', 'trans', 'opacities']  # ← 여기도
        render_result = self.render_once(rays, query_keys=query_keys,
                                         sampling_requires_grad=self.need_to_update_occ(self.global_iter_step_app),
                                         geo_inference=True)

        if (render_result is None) or (not render_result['is_valid']):
            self.global_iter_step_app += 1
            return

        color_loss_weight = train_conf.color_loss_weight #1
        if color_loss_weight > eps:
            pred_colors = render_result['rgb']

            color_loss = F.smooth_l1_loss(pred_colors, gt_colors, beta=5e-2, reduction='mean')
            self.writer.add_scalar('nerf_loss/color_loss', color_loss, self.global_iter_step_app)
            loss = loss + color_loss * color_loss_weight

        if grad_scaler is None:
            loss.backward()
        else:
            grad_scaler.scale(loss).backward()
        optimizer.step()

        self.writer.add_scalar('others/lr_app', optimizer.param_groups[0]['lr'], self.global_iter_step_app)

        self.global_iter_step_app += 1


    def update_lr(self, optimizer, optim_conf, progress):
        # Update scene optimizer lr
        if progress < optim_conf.peak_at:
            local_progress = progress / optim_conf.peak_at
            lr = optim_conf.peak_lr * local_progress + optim_conf.init_lr * (1. - local_progress)
        else:
            local_progress = (progress - optim_conf.peak_at) / (1. - optim_conf.peak_at)
            lr_factor = (np.cos(local_progress * np.pi) + 1.) * .5 * (1. - optim_conf.lr_alpha) + optim_conf.lr_alpha
            lr = optim_conf.peak_lr * lr_factor

        for p in optimizer.param_groups:
            p['lr'] = lr

    def to_bounded_rays(self, rays):
        rays_o = rays.o
        rays_d = rays.d
        batch_size = len(rays_o)
        near = 1e-2 * torch.ones([batch_size, 1])
        far  = 1 * torch.ones([batch_size, 1])
        return BoundedRays(rays_o, rays_d, near, far)

    def get_pano_visibility_mask(self, sup_pool, rays):
        distance = self.render(rays, query_keys=['distance'])['distance'].squeeze()
        write_image(pjoin(self.base_exp_dir, f'distance.png'), colorize_singel_channel_image(distance))
        height, width = distance.shape
        pts = rays.o + rays.d * distance[..., None]

        mask = torch.zeros([height, width, 1])

        for pano_idx in range(len(sup_pool.sup_infos)):
            sup_info = sup_pool.sup_infos[pano_idx]
            sup_distance_map = sup_info.distance_map * sup_info.mask
            write_image(pjoin(self.base_exp_dir, f'idx_{pano_idx}_sup_distance_map.png'), colorize_singel_channel_image(sup_distance_map))
            new_dirs = apply_rot(pts - sup_info.pose[:3, 3], sup_info.pose[:3, :3].T)
            new_distances = torch.linalg.norm(new_dirs, 2, -1, True)
            write_image(pjoin(self.base_exp_dir, f'idx_{pano_idx}_new_distance.png'), colorize_singel_channel_image(new_distances))
            new_dirs /= new_distances
            proj_coords = direction_to_img_coord(new_dirs)
            sample_coords = img_coord_to_sample_coord(proj_coords)
            proj_distances = F.grid_sample(sup_distance_map[None].permute(0, 3, 1, 2), sample_coords[None],
                                        padding_mode='border')
            proj_distances = proj_distances[0].permute(1, 2, 0)
            write_image(pjoin(self.base_exp_dir, f'idx_{pano_idx}_proj_distances.png'), colorize_singel_channel_image(proj_distances))
            # bias = (new_distances - proj_distances).clip(0., None) / (new_distances / 256.0).clip(2.5e-3, None)
            # bias = torch.exp(-bias * bias * .5)
            # bias = (new_distances - proj_distances).clip(0., None) < (1 / 256.)
            bias = new_distances < proj_distances + 1 / 256.
            mask.clamp_(min=bias, max=None)

        l_size = (9, 9)
        s_size = (5, 5)

        kernel_l = cv.getStructuringElement(cv.MORPH_ELLIPSE, l_size)
        kernel_s = cv.getStructuringElement(cv.MORPH_ELLIPSE, s_size)
        kernel_l = torch.from_numpy(kernel_l).to(torch.float32).to(mask.device)
        kernel_s = torch.from_numpy(kernel_s).to(torch.float32).to(mask.device)

        mask = (mask[None, :, :, :] > 0.5).float()
        mask = mask.permute(0, 3, 1, 2)
        mask = dilation(mask, kernel=kernel_s)
        mask = erosion(mask, kernel=kernel_l)

        return mask.permute(0, 2, 3, 1).contiguous().squeeze()

    def need_to_update_occ(self, iter_step):
        # return (iter_step % 4 == 0) and self.nerf.training
        return True

    def update_occ_grid(self, iter_step, trans):
        if self.need_to_update_occ(iter_step):
            self.estimator._update(trans=trans.detach())

    def load_state_dict(self, state_dict):
        # self.optimizer.load_state_dict(state_dict['optimizer'])
        self.renderer.load_state_dict(state_dict['render'])
        self.nerf.load_state_dict(state_dict['nerf'])
        self.estimator.load_state_dict(state_dict['estimator'])

    def state_dict(self):
        return {
            # 'optimizer': self.optimizer.state_dict(),
            'render': self.renderer.state_dict(),
            'nerf': self.nerf.state_dict(),
            'estimator': self.estimator.state_dict(),
        }

    def set_train(self):
        self.nerf.train()
        self.estimator.train()
        if isinstance(self.estimator, PropNetEstimator):
            for p in self.prop_networks:
                p.train()
        self.renderer.train()

    def set_eval(self):
        self.nerf.eval()
        self.estimator.eval()
        if isinstance(self.estimator, PropNetEstimator):
            for p in self.prop_networks:
                p.eval()
        self.renderer.eval()

    @torch.no_grad()
    def diagnose_culling(self, sup_pool):
        # 1. GT가 있는 ray들을 가져온다 (전체 픽셀)
        rays, gt_colors, gt_depths, gt_normals = sup_pool.rand_ray_color_data(
            batch_size=8192, rand_mode='by_all_pixels')
        rays = self.to_bounded_rays(rays)
        rays_o, rays_d, near, far = rays.collapse()

        # 2. estimator가 각 ray에 대해 sample interval을 만드는지 확인
        #    renderer.render를 타되, ray_indices로 "샘플 잡힌 ray" 집합을 구한다
        render_result = self.renderer.render(self.nerf, self.estimator,
                                            rays_o, rays_d, near, far,
                                            geo_inference=True)
        ray_indices = render_result['ray_indices']
        n_rays = rays_o.shape[0]
        has_sample = torch.zeros(n_rays, dtype=torch.bool)
        has_sample[ray_indices.unique()] = True

        # 3. GT상 표면이 near~far 안에 있는 ray (= 원래 샘플이 잡혀야 하는 ray)
        gt_valid = (gt_depths.squeeze() > near.squeeze()) & (gt_depths.squeeze() < far.squeeze())

        # 4. 교차: GT엔 표면이 있는데 샘플이 안 잡힌 ray = false-negative culling
        culled_but_real = gt_valid & (~has_sample)
        print('GT valid rays:', gt_valid.sum().item())
        print('falsely culled :', culled_but_real.sum().item())

        # culled_but_real을 (H, W)로 reshape해서 시각화
        write_image(pjoin(self.base_exp_dir, 'culled_mask.png'),
                    culled_but_real.reshape(self.dataset.height, self.dataset.width, 1).float() * 255.)
        return culled_but_real

    @torch.no_grad()
    def check_grid_coverage(self, sup_pool, occ_res=256):
        # GT 표면 점 구름 (gen_occ_grid가 내부적으로 쓰는 occ_pts 활용)
        pre_grid, occ_pts = sup_pool.gen_occ_grid(res=occ_res)

        # occ_pts를 voxel 인덱스로 변환 (occ_eval_fn과 동일한 매핑)
        x = (occ_pts.clip(-0.999, 0.999) * .5 + .5) * occ_res
        x = x.to(torch.int64)
        idx = x[...,0]*occ_res*occ_res + x[...,1]*occ_res + x[...,2]

        # estimator의 현재 binary grid와 대조
        occ_binary = self.estimator.binaries.reshape(-1)   # nerfacc 내부 buffer
        covered = occ_binary[idx]
        print('GT surface voxels:', len(idx))
        print('covered by grid  :', covered.sum().item())
        print('missing (culled) :', (~covered).sum().item())

    @torch.no_grad()
    def diagnose_culling_re(self, sup_pool, H, W):
        self.set_eval()

        # --- GT distance map 가져오기 ---
        dist_map = sup_pool.sup_infos[0].distance_map
        print('distance_map shape:', tuple(dist_map.shape))
        gt_depths = dist_map.reshape(-1).to(torch.float32)   # ← 이 줄이 핵심, 반드시 살리기

        # --- 전 픽셀 ray 생성 (identity pose, 1.png와 동일 조건) ---
        rays = gen_pano_rays(torch.eye(4), H, W)
        rays_o, rays_d = rays.collapse()
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        n_rays = rays_o.shape[0]
        print('n_rays:', n_rays, '| gt_depths:', gt_depths.numel())
        assert n_rays == gt_depths.numel(), \
            'ray 개수와 GT depth 개수 불일치 — H,W 또는 평탄화 순서 확인 필요'

        # --- 샘플이 잡힌 ray 집합 (배치로 끊어서 render_once 재사용) ---
        has_sample = torch.zeros(n_rays, dtype=torch.bool, device=rays_o.device)
        opacity_full = torch.zeros(n_rays, device=rays_o.device)   
        batch_size = 16384
        valid_batches = 0
        invalid_batches = 0
        for start in range(0, n_rays, batch_size):
            end = min(start + batch_size, n_rays)
            sub = Rays(rays_o[start:end], rays_d[start:end])
            rr = self.render_once(
                sub,
                query_keys=['rgb', 'distance', 'weights', 't_starts', 't_ends', 'trans', 'ray_indices', 'opacities'],
                # query_keys=['rgb', 'distance', 'weights', 't_starts', 't_ends', 'trans', 'ray_indices'],
                geo_inference=True,
            )
            if (rr is None) or (not rr['is_valid']):
                invalid_batches += 1
                continue
            valid_batches += 1
            local_idx = rr['ray_indices'].unique()
            has_sample[start + local_idx] = True
            opacity_full[start:end] = rr['opacities'].reshape(-1)

        print('valid batches:', valid_batches, '| invalid batches:', invalid_batches)

        # --- near/far 안에 GT 표면이 있는 ray ---
        near, far = 1e-2, 1.0   # to_bounded_rays와 동일
        gt_valid = (gt_depths > near) & (gt_depths < far)

        # --- 교차: GT엔 표면 있는데 샘플 안 잡힘 ---
        culled_but_real = gt_valid & (~has_sample)

        print('total rays     :', n_rays)
        print('GT valid rays  :', gt_valid.sum().item())
        print('has_sample     :', has_sample.sum().item())
        print('falsely culled :', culled_but_real.sum().item())

        # --- 시각화: 검은 얼룩과 대조용 ---
        write_image(pjoin(self.base_exp_dir, 'culled_mask.png'),
                    culled_but_real.reshape(H, W, 1).float() * 255.)
        write_image(pjoin(self.base_exp_dir, 'gt_valid_mask.png'),
                    gt_valid.reshape(H, W, 1).float() * 255.)
        write_image(pjoin(self.base_exp_dir, 'has_sample_mask.png'),
                    has_sample.reshape(H, W, 1).float() * 255.)
        # GT depth가 ray 순서와 맞는지 눈으로 확인용 (distance_vis.png와 같은 모양이어야 함)
        write_image(pjoin(self.base_exp_dir, 'gt_depth_check.png'),
                    colorize_single_channel_image(
                        (gt_depths.min() + 1e-6) / (gt_depths.reshape(H, W, 1) + 1e-6)))
        write_image(pjoin(self.base_exp_dir, 'opacity_map.png'),
            (1-opacity_full.reshape(H, W, 1)) * 255.)

        # culled ray들의 GT depth가 실제로 far 근처인지 확인
        culled_depths = gt_depths[culled_but_real]
        if culled_depths.numel() > 0:
            print('[culled depth] min/max/mean:',
                culled_depths.min().item(),
                culled_depths.max().item(),
                culled_depths.mean().item())
            # far(=1.0 또는 1.8) 근처 비율
            for thr in [0.9, 0.95, 1.0, 1.5]:
                print(f'  frac > {thr}:', (culled_depths > thr).float().mean().item())

        # return culled_but_real
    @torch.no_grad()
    def diagnose_gray_vs_occ(self, sup_pool, pano_idx=0, res=256,
                             opacity_thre=0.5, axis=2, num_slices=256,
                             upscale=4, save_dir=None):
        """
        회색 픽셀(렌더 opacity 낮음 & GT 유효)을 GT distance로 백프로젝션해
        occ grid 슬라이스 위에 시안으로 오버레이.
        시안이 빨강(occupied) 아닌 칸에 앉으면 = 그 voxel이 culled.
        """
        import os
        import numpy as np
        import cv2 as cv

        self.set_eval()
        if save_dir is None:
            save_dir = pjoin(self.base_exp_dir, f'gray_vs_occ_{res}')
        os.makedirs(save_dir, exist_ok=True)

        # ---------- 1. 회색 픽셀 → GT 표면점 ----------
        info = sup_pool.sup_infos[pano_idx]
        H, W = info.distance_map.shape[:2]

        rays = gen_pano_rays(info.pose, H, W)
        rr = self.render(rays, query_keys=['distance', 'opacities'])
        opac = rr['opacities'].reshape(H, W)
        # breakpoint()
        unique_values = torch.unique(opac)
        print(unique_values)
        print("unique 개수:", unique_values.numel())

        unique_values, counts = torch.unique(opac, return_counts=True)

        for value, count in zip(unique_values, counts):
            print(f"value={value.item():.15f}, count={count.item()}")

        output_path = pjoin(self.base_exp_dir, "opac_unique_values.txt")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"unique 개수: {unique_values.numel()}\n")
            f.write("=" * 40 + "\n")

            for value, count in zip(unique_values, counts):
                f.write(
                    f"value={value.item():.8f}, count={count.item()}\n"
                )

        d_gt = info.distance_map.reshape(H, W).to(opac.device)
        # gt_mask = (info.mask.reshape(H, W) > 0.5)
        # gray = (opac < opacity_thre) & gt_mask                # 회색 & GT엔 표면 있음
        gray = (opac < opacity_thre) 


        ro, rd = rays.collapse()
        gray_pts = (ro + rd * d_gt[..., None])[gray]          # (N,3) world
        print(f'[gray] gray & GT-valid pixels: {gray.sum().item()} / {opac.sum().item()}')
        

        # ---------- 2. occ grid 받아오기 (결정론적이라 재호출 OK) ----------
        occ_grid, _ = sup_pool.gen_occ_grid(res=res)
        occ_3d = occ_grid.detach().cpu().numpy().reshape(res, res, res)  # [X,Y,Z]

        # ---------- 3. voxel 매핑 (gen_occ_grid와 '동일한' 식) ----------
        # gen_occ_grid: ((p.clip(-0.999,0.999)*.5+.5)*res).to(int64)
        gp = gray_pts.detach().cpu().numpy()
        in_b = np.all((gp >= -1.0) & (gp < 1.0), axis=1)
        gray_vox = np.clip(
            ((np.clip(gp, -0.999, 0.999) * 0.5 + 0.5) * res).astype(np.int64),
            0, res - 1
        )[in_b]                                                # [N,3] (x,y,z)

        # ---------- 3.5 정량 판정: 시안이 empty 칸에 앉는 비율 ----------
        gidx = gray_vox[:, 0] * res * res + gray_vox[:, 1] * res + gray_vox[:, 2]
        occ_flat = occ_grid.detach().cpu().numpy().reshape(-1)
        on_occ = occ_flat[gidx] > 0
        n = len(gidx)
        print(f'[gray] on OCCUPIED voxel : {int(on_occ.sum())} / {n} '
              f'({100.0*on_occ.mean():.1f}%)')
        print(f'[gray] on EMPTY   voxel : {int((~on_occ).sum())} / {n} '
              f'({100.0*(~on_occ).mean():.1f}%)')

        # ---------- 4. 슬라이스 렌더 (빨강/초록/시안 3색) ----------
        # gray_vox의 각 점이 occupied인지 = 이미 3.5에서 구한 on_occ
        gv_occ = gray_vox[on_occ]        # 초록: gray_pt가 occupied 칸에 앉음
        gv_emp = gray_vox[~on_occ]       # 시안: gray_pt가 empty 칸에 앉음 (← 문제 지점)

        axis_name = ['x', 'y', 'z'][axis]
        proj = {0: [1, 2], 1: [0, 2], 2: [0, 1]}[axis]
        slice_indices = np.unique(
            np.linspace(0, res - 1, num_slices).round().astype(int))
        slice_imgs = []

        # def paint(img, coords, color):
        #     if len(coords) == 0:
        #         return
        #     for dy in (-1, 0, 1):          # 3x3으로 두껍게 (1픽셀이라 안 보이는 문제 방지)
        #         for dx in (-1, 0, 1):
        #             yy = np.clip(coords[:, 0] + dy, 0, img.shape[0] - 1)
        #             xx = np.clip(coords[:, 1] + dx, 0, img.shape[1] - 1)
        #             img[yy, xx] = color

        def paint(img, coords, color):
            if len(coords) == 0:
                return
            yy = np.clip(coords[:, 0], 0, img.shape[0] - 1)
            xx = np.clip(coords[:, 1], 0, img.shape[1] - 1)
            img[yy, xx] = color

        for k in slice_indices:
            if axis == 0:
                occ_slice = occ_3d[k, :, :]
            elif axis == 1:
                occ_slice = occ_3d[:, k, :]
            else:
                occ_slice = occ_3d[:, :, k]

            img = np.zeros((*occ_slice.shape, 3), dtype=np.uint8)
            img[occ_slice > 0] = (0, 0, 255)                       # ① occupied = red (BGR)

            # 이 슬라이스(k)에 걸린 gray_pt만 골라 색칠
            sel_g = gv_occ[gv_occ[:, axis] == k][:, proj]          # ② green
            sel_c = gv_emp[gv_emp[:, axis] == k][:, proj]          # ③ cyan
            paint(img, sel_g, (0, 255, 0))                         # green (BGR)
            paint(img, sel_c, (255, 255, 0))                       # cyan  (BGR), 맨 위

            if upscale > 1:
                img = cv.resize(img, None, fx=upscale, fy=upscale,
                                interpolation=cv.INTER_NEAREST)
            cv.putText(img, f"{axis_name}={k}", (5, 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv.imwrite(pjoin(save_dir, f"slice_{axis_name}_{k:04d}.png"), img)
            slice_imgs.append(img)

        # ---------- 5. montage ----------
        if slice_imgs:
            cnt = len(slice_imgs)
            cols = int(np.ceil(np.sqrt(cnt)))
            rows = int(np.ceil(cnt / cols)) 
            h, w = slice_imgs[0].shape[:2]
            canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
            for i, im in enumerate(slice_imgs):
                r, c = divmod(i, cols)
                canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
            cv.imwrite(pjoin(save_dir, f"montage_{axis_name}.png"), canvas)

        print(f'[gray] saved {len(slice_indices)} slices to: {save_dir}')
        return gray_pts
    
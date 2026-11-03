import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2 as cv
import kornia
from kornia.morphology import erosion, dilation

from utils.camera_utils import *
from utils.utils import write_image
from icecream import ic
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

class SupInfo(nn.Module):
    # Support:
    # Supervision of color and distance.
    # Output occlusion info given points
    def __init__(self):
        super().__init__()

    def get_occulsion_mask(self, pts):
        raise NotImplementedError


class PanoSupInfo(SupInfo):
    def __init__(self, pose, mask, color_map, distance_map, normal_map=None, factor=1):
        '''
        :param pose: [4, 4]
        :param mask: [H, W, 1] or [H, W]
        :param color_map: [H, W, 3]
        :param distance_map: [H, W, 1] or [H, W]
        :param normal_map: [H, W, 3]
        :param factor:
        '''
        super().__init__()
        device = color_map.device
        height, width, _ = color_map.shape

        if distance_map is None:
            distance_map = torch.ones([height, width, 1], device=color_map.device)
        else:
            distance_map = distance_map.squeeze()[..., None]
        mask = mask.squeeze()[..., None]

        has_normal_map = True
        if normal_map is None:
            has_normal_map = False
            normal_map = torch.zeros([height, width, 3], device=color_map.device)

        assert color_map.shape[-1] == 3 and distance_map.shape[-1] == 1
        self.register_buffer('pose', pose)

        if factor != 1:
            factor = int(factor)
            height = height // factor
            width = width // factor

            color_map = cv.resize(color_map.cpu().numpy(), (width, height), interpolation=cv.INTER_AREA)
            distance_map = cv.resize(distance_map.cpu().numpy(), (width, height), interpolation=cv.INTER_AREA)
            normal_map = cv.resize(normal_map.cpu().numpy(), (width, height), interpolation=cv.INTER_AREA)

            color_map = torch.from_numpy(color_map).to(device)
            distance_map = torch.from_numpy(distance_map).to(device)
            normal_map = torch.from_numpy(normal_map).to(device)

        self.height, self.width = height, width
        if mask is None:
            mask = torch.ones_like(distance_map, dtype=torch.bool)
        else:
            mask = (mask > .5)

        mask = mask & (distance_map > 1e-5)
        self.register_buffer('mask_raw', mask.clone())

        x_laplacian = kornia.filters.laplacian(distance_map[None].permute(0, 3, 1, 2), kernel_size=3)
        edge_mask = (x_laplacian.abs() < 0.01).float()
        edge_mask = erosion(edge_mask, kernel=torch.ones(3, 3))
        edge_mask = dilation(edge_mask, kernel=torch.ones(3, 3))

        mask = mask & (edge_mask[0] > .5).permute(1, 2, 0)

        if has_normal_map:
            pano_dirs = -img_coord_to_pano_direction(img_coord_from_hw(height, width))
            normal_cos = (pano_dirs * normal_map).sum(-1, True).clip(0., 1.)
            mask = mask & (normal_cos > 0.15)

        self.register_buffer('color_map', color_map)
        self.register_buffer('distance_map', distance_map)
        self.register_buffer('normal_map', normal_map)
        self.register_buffer('mask', mask)

        # self.sup_colors = None
        # self.sup_distances = None
        # self.sup_normals = None
        # self.sup_rays = None
        self.update_sup_info()

    def update_sup_info(self):
        pose, height, width = self.pose, self.height, self.width
        img_coords = torch.meshgrid(torch.linspace(.5 / height, 1. - .5 / height, height),
                                    torch.linspace(.5 / width,  1. - .5 / width,  width),
                                    indexing='ij')
        # breakpoint()
        dirs = img_coord_to_pano_direction(torch.stack(img_coords, -1))
        dirs = apply_rot(dirs, self.pose[:3, :3])

        positions = self.pose[None, None, :3, 3].repeat(height, width, 1)

        sup_indices = torch.where(self.mask[..., 0] > 0.5)
        sup_colors, sup_distances, sup_normals =\
            self.color_map[sup_indices], self.distance_map[sup_indices], self.normal_map[sup_indices]
        sup_dirs, sup_positions = dirs[sup_indices], positions[sup_indices]

        self.register_buffer('sup_colors', sup_colors)
        self.register_buffer('sup_distances', sup_distances)
        self.register_buffer('sup_normals', sup_normals)
        self.register_buffer('sup_dirs', sup_dirs)
        self.register_buffer('sup_positions', sup_positions)

        self.sup_rays = Rays(sup_positions, sup_dirs)
        #ray 확인
        # breakpoint()

    def get_pers_patch_data(self, res, fov, from_masked_region=True):
        local_pers_rays_d = cam_rays_cam_space(res, res, fovy=fov)
        if not from_masked_region:
            to_vec = torch.randn(3)
            to_vec /= torch.linalg.norm(to_vec, 2, -1, True)
        else:
            coords = torch.stack([
                (self.sup_indices[0] + .5) / self.height,
                (self.sup_indices[1] + .5) / self.width
            ], dim=-1)
            dirs = img_coord_to_pano_direction(coords)
            idx = np.random.randint(0, len(dirs))
            to_vec = dirs[idx]
            to_vec /= torch.linalg.norm(to_vec, 2, -1, True)

        local_rots = look_at(to_vec[None])[0]
        # local_pers_rays_d = torch.matmul(local_rots[None, None, :, :], local_pers_rays_d[:, :, :, None])[..., 0]
        local_pers_rays_d = apply_rot(local_pers_rays_d, local_rots)
        img_coords = direction_to_img_coord(local_pers_rays_d)
        sampled_coords = img_coord_to_sample_coord(img_coords)
        colors = F.grid_sample(self.color_map[None].permute(0, 3, 1, 2), sampled_coords[None])[0].permute(1, 2, 0)
        rays = Rays(torch.zeros_like(local_pers_rays_d) + self.pose[:3, 3], apply_rot(local_pers_rays_d, self.pose[:3, :3]))
        return { 'colors': colors, 'rays': rays }

    def query_color_at_pts(self, pts):
        """PanoGRF-style cross-view lookup: project world-space 3D points onto this
        panorama and sample color via bilinear interpolation.

        Mirrors the geo_check projection logic:
          1. Compute direction from this camera center to each point (world-space)
          2. Rotate to camera-local frame (R_c2w^T, same as geo_check)
          3. Map to panoramic image coords and sample color
          4. Weight by mask validity + depth consistency (visible if pt is in front
             of the reference surface)

        :param pts: [N, 3] world-space 3D points
        :return:
            colors:  [N, 3] bilinearly-sampled colors from this panorama
            weights: [N]    per-point validity weight (0 = invisible)
        """
        cam_pos = self.pose[:3, 3]  # [3]

        # Direction from camera center to each point, in world space
        dirs_world = pts - cam_pos.unsqueeze(0)          # [N, 3]
        distances  = torch.linalg.norm(dirs_world, 2, -1)  # [N]
        dirs_world_norm = dirs_world / (distances.unsqueeze(-1) + 1e-8)

        # Rotate to camera-local frame (R_c2w^T)
        dirs_local = apply_rot(dirs_world_norm, self.pose[:3, :3].T)  # [N, 3]

        # Panoramic image coordinates → grid_sample coordinates
        img_coords    = direction_to_img_coord(dirs_local)          # [N, 2] in [0,1]
        sample_coords = img_coord_to_sample_coord(img_coords)        # [N, 2] in [-1,1]
        # grid_sample expects [N_batch, H_out, W_out, 2]; use [1, 1, N, 2]
        sample_grid = sample_coords.unsqueeze(0).unsqueeze(0)        # [1, 1, N, 2]

        # Sample color
        color_chw = self.color_map.permute(2, 0, 1).unsqueeze(0)     # [1, 3, H, W]
        sampled_colors = F.grid_sample(
            color_chw, sample_grid,
            align_corners=False, mode='bilinear', padding_mode='border',
        )                                                             # [1, 3, 1, N]
        colors = sampled_colors[0, :, 0, :].T                        # [N, 3]

        # Sample validity mask
        mask_chw = self.mask.float().permute(2, 0, 1).unsqueeze(0)   # [1, 1, H, W]
        sampled_mask = F.grid_sample(
            mask_chw, sample_grid,
            align_corners=False, mode='bilinear', padding_mode='zeros',
        )                                                             # [1, 1, 1, N]
        valid = sampled_mask[0, 0, 0, :]                             # [N]

        # Sample reference depth (masked) for depth-consistency check
        dist_chw = (self.distance_map * self.mask.float()).permute(2, 0, 1).unsqueeze(0)
        sampled_dist = F.grid_sample(
            dist_chw, sample_grid,
            align_corners=False, mode='bilinear', padding_mode='zeros',
        )                                                             # [1, 1, 1, N]
        ref_distances = sampled_dist[0, 0, 0, :]                     # [N]

        # A point is depth-consistent when it lies at or in front of the surface
        # (allowing a small margin of 0.05 in normalised scene units)
        depth_consistent = (distances <= ref_distances + 0.05).float()

        weights = valid * depth_consistent
        return colors, weights

    def set_after_reload(self):
        self.sup_rays = Rays(self.sup_positions, self.sup_dirs)


class SupInfoPool:
    def __init__(self):
        super().__init__()
        self.sup_infos = list()
        self.all_sup_colors = None
        self.all_sup_rays = None
        self.all_sup_distances = None
        self.all_sup_normals = None

        # 실행 시작 시 로그 파일 초기화
        for fname in ("sup_info_pool_rays.txt", "ray_origins.txt"):
            open(os.path.join(os.getcwd(), fname), "w").close()

    def register_sup_info(self, pose, mask, rgb, distance, normal=None, log=False):
        self.sup_infos.append(PanoSupInfo(pose=pose, mask=mask, color_map=rgb, distance_map=distance, normal_map=normal))
        img_idx = len(self.sup_infos) - 1
        cur_info = self.sup_infos[-1]

        if self.all_sup_colors is None:
            self.all_sup_colors = self.sup_infos[0].sup_colors
            self.all_sup_rays = self.sup_infos[0].sup_rays
            self.all_sup_distances = self.sup_infos[0].sup_distances
            self.all_sup_normals = self.sup_infos[0].sup_normals
        else:
            self.all_sup_colors = torch.cat([self.all_sup_colors, self.sup_infos[-1].sup_colors], 0)
            self.all_sup_rays = cat_rays([self.all_sup_rays, self.sup_infos[-1].sup_rays])
            self.all_sup_distances = torch.cat([self.all_sup_distances, self.sup_infos[-1].sup_distances], 0)
            self.all_sup_normals = torch.cat([self.all_sup_normals, self.sup_infos[-1].sup_normals], 0)

        # if log:
        #     with open(os.path.join(os.getcwd(), "ray_origins.txt"), "a") as f:
        #         f.write(f"===== image {img_idx} =====\n")
        #         f.write(f"pose[:3, 3] (camera origin): {pose[:3, 3].tolist()}\n")
        #         f.write(f"sup_positions shape: {cur_info.sup_positions.shape}\n")
        #         f.write(f"sup_positions[0] (first ray.o): {cur_info.sup_positions[0].tolist()}\n")
        #         unique_origins = cur_info.sup_positions.unique(dim=0)
        #         f.write(f"unique origins in this image: {unique_origins.shape[0]} (should be 1)\n")
        #         f.write(f"unique origin value: {unique_origins.tolist()}\n\n")

        #     with open(os.path.join(os.getcwd(), "sup_info_pool_rays.txt"), "a") as f:
        #         f.write(f"****** sup info pool rays (image {img_idx}) *******\n")
        #         f.write(f"this image rays o shape: {cur_info.sup_rays.o.shape}\n")
        #         f.write(f"this image rays d shape: {cur_info.sup_rays.d.shape}\n")
        #         f.write(f"o type: {type(cur_info.sup_rays.o)}\n")
        #         f.write(f"d type: {type(cur_info.sup_rays.d)}\n")
        #         f.write(f"n_rays this image: {len(cur_info.sup_colors)}\n")
        #         f.write(f"n_rays total pool: {len(self.all_sup_colors)}\n")



    def register_sup_info_by_pts(self, pose, colors, pts):
        H, W, _ = pts.shape
        pts = pts - pose[:3, 3][None, None, :]
        pts = apply_rot(pts, torch.linalg.inv(pose[:3, :3]))
        new_d = torch.linalg.norm(pts, 2, -1, False)

        # normalize: /depth
        new_dirs = pts / new_d.reshape(H, W, 1)
        new_depth = torch.zeros(new_d.shape)
        img = torch.zeros(pts.shape)

        # backward: 3d coordinate to pano image
        # [x, y, z] = new_coord[..., 0], new_coord[..., 1], new_coord[..., 2]

        idx = torch.where(new_d > 0)

        # theta: horizontal angle, phi: vertical angle
        # theta = torch.zeros(y.shape)
        # phi = np.zeros(y.shape)
        # x1 = np.zeros(z.shape)
        # y1 = np.zeros(z.shape)

        img_coord = direction_to_img_coord(new_dirs)
        x = torch.floor(img_coord[..., 0] * H).to(torch.int64)
        y = torch.floor(img_coord[..., 1] * W).to(torch.int64)

        # Mask out
        mask = (new_d > 0) & (H > x) & (x > 0) & (W > y) & (y > 0)
        x = x[torch.where(mask)]
        y = y[torch.where(mask)]
        new_d = new_d[mask]
        colors = colors[mask]
        reorder = torch.argsort(-new_d)
        x = x[reorder]
        y = y[reorder]
        new_d = new_d[reorder]
        colors = colors[reorder]
        # Assign
        new_depth[x, y] = new_d
        img[x, y] = colors

        depth_margin = 4
        for i in tqdm(range(depth_margin, H, 2)):
            for j in range(depth_margin, W, 2):
                x_l = max(0, i - depth_margin)
                x_r = min(H, i + depth_margin)
                y_l, y_r = max(0, j - depth_margin), min(W, j + depth_margin)

                index = torch.where(new_depth[x_l:x_r, y_l:y_r] > 0)
                if len(index[0]) == 0: continue
                mean = torch.median(new_depth[x_l:x_r, y_l:y_r][index])  # median
                target_index = torch.where(new_depth[x_l:x_r, y_l:y_r] > mean * 1.3)

                if len(target_index[0]) > depth_margin ** 2 // 2:
                    # reduce block size
                    img[x_l:x_r, y_l:y_r][target_index] = 0  # np.array([255.0, 0.0, 0.0])
                    new_depth[x_l:x_r, y_l:y_r][target_index] = 0

        mask = (new_depth != 0).float()

        self.register_sup_info(pose, mask, img, distance=None, normal=None)


    def query_ibr_colors(self, pts):
        """PanoGRF-style Image-Based Rendering: aggregate colors from all reference
        panoramas at the given world-space 3D positions.

        For each point, each reference view contributes a color weighted by its
        mask validity and depth consistency (analogous to PanoGRF's visibility-
        weighted aggregation in DefaultAggregationNet).

        :param pts: [N, 3] world-space 3D points (typically the rendered surface
                    position for each training ray: weighted avg of sample points)
        :return:
            ibr_colors:    [N, 3] visibility-weighted average color across views
            total_weights: [N]    sum of per-view weights (0 → point invisible)
        """
        all_colors  = []
        all_weights = []

        for sup_info in self.sup_infos:
            colors, weights = sup_info.query_color_at_pts(pts)
            all_colors.append(colors)
            all_weights.append(weights)

        all_colors  = torch.stack(all_colors,  dim=0)   # [V, N, 3]
        all_weights = torch.stack(all_weights, dim=0)   # [V, N]

        total_weights = all_weights.sum(0)               # [N]

        # Normalise weights across views and compute weighted-average color
        safe_weights = all_weights / (total_weights.unsqueeze(0) + 1e-8)  # [V, N]
        ibr_colors   = (all_colors * safe_weights.unsqueeze(-1)).sum(0)   # [N, 3]

        return ibr_colors, total_weights

    def rand_ray_color_data(self, batch_size, pano_idx=-1, rand_mode='by_all_pixels'):
        assert rand_mode in ['by_all_pixels', 'only_first', 'only_last']

        if rand_mode == 'by_all_pixels':
            sup_colors = self.all_sup_colors
            sup_rays = self.all_sup_rays
            sup_distances = self.all_sup_distances
            sup_normals = self.all_sup_normals
            assert len(sup_colors) == len(sup_rays) == len(sup_distances)
        elif rand_mode == 'only_first':
            sup_colors = self.sup_infos[0].sup_colors
            sup_rays = self.sup_infos[0].sup_rays
            sup_distances = self.sup_infos[0].sup_distances
            sup_normals = self.sup_infos[0].sup_normals
        else:
            sup_colors = self.sup_infos[-1].sup_colors
            sup_rays = self.sup_infos[-1].sup_rays
            sup_distances = self.sup_infos[-1].sup_distances
            sup_normals = self.sup_infos[-1].sup_normals

        max_ray_idx = len(sup_colors)
        indices = torch.randint(0, max_ray_idx, (batch_size,))

        return sup_rays[indices], sup_colors[indices], sup_distances[indices], sup_normals[indices]

    def geo_check(self, rays, distances):
        '''
        :param rays:
        :param distances:
        :return: mask, 1 -> OK! 0 -> conflict!
        '''
        pts = rays.o + rays.d * distances.squeeze()[..., None]
        height, width = pts.shape[:2]
        mask = torch.ones([height, width, 1])

        for pano_idx in range(len(self.sup_infos)):
            sup_info = self.sup_infos[pano_idx]
            sup_distance_map = sup_info.distance_map * sup_info.mask.float()

            new_dirs = apply_rot(pts - sup_info.pose[:3, 3], sup_info.pose[:3, :3].T)
            new_distances = torch.linalg.norm(new_dirs, 2, -1, True)
            new_dirs /= new_distances
            proj_coords = direction_to_img_coord(new_dirs)
            sample_coords = img_coord_to_sample_coord(proj_coords)
            proj_distances = F.grid_sample(sup_distance_map[None].permute(0, 3, 1, 2), sample_coords[None],
                                           padding_mode='border')
            proj_distances = proj_distances[0].permute(1, 2, 0)
            # bias = (proj_distances - new_distances).clip(0., None) / (new_distances / 256.0).clip(2.5e-3, None)
            # bias = torch.exp(-bias * bias * .5)
            # bias = ((proj_distances - new_distances).clip(0., None) < (1. / 512.)).float()
            bias = (proj_distances < new_distances).float()
            mask.clamp_(min=None, max=bias)

        l_size = (9, 9)
        s_size = (3, 3)

        kernel_l = cv.getStructuringElement(cv.MORPH_ELLIPSE, l_size)
        kernel_s = cv.getStructuringElement(cv.MORPH_ELLIPSE, s_size)
        kernel_l = torch.from_numpy(kernel_l).to(torch.float32).to(mask.device)
        kernel_s = torch.from_numpy(kernel_s).to(torch.float32).to(mask.device)

        mask = (mask[None, :, :, :] > 0.5).float()
        mask = mask.permute(0, 3, 1, 2)
        mask = dilation(mask, kernel=kernel_s)
        mask = erosion(mask, kernel=kernel_l)

        return mask.permute(0, 2, 3, 1).contiguous().squeeze()

    def gen_occ_grid(self, res):
        rays_o, rays_d = self.all_sup_rays.collapse()
        dis = self.all_sup_distances
        pts = rays_o + rays_d * dis.squeeze()[..., None]
        occ_grid = torch.zeros([res * res * res], dtype=torch.uint8)

        # if not hasattr(self, "ply_points"):
        #     raise RuntimeError("Call load_geometry_ply() first")

        # # 저장된 PLY 좌표를 그대로 사용
        # pts = self.ply_points

        # occ_grid = torch.zeros(
        #     [res * res * res],
        #     dtype=torch.uint8,
        #     device=pts.device,
        # )
        # shift = 1. / res
        # xx, yy, zz = torch.meshgrid(
        #     torch.linspace(-shift, shift, 3),
        #     torch.linspace(-shift, shift, 3),
        #     torch.linspace(-shift, shift, 3)
        # )
        shift = 1. / res
        offsets = torch.linspace(
            -shift, shift, 3,
            device=pts.device,
            dtype=pts.dtype,
        )
        xx, yy, zz = torch.meshgrid(
            offsets, offsets, offsets, indexing="ij"
        )
        shift_xyzs = torch.stack([xx, yy, zz], -1).reshape(-1, 3)
        for shift_xyz in shift_xyzs:
            shifted = shift_xyz[None, :] + pts
            shifted = ((shifted.clip(-0.999, 0.999) * .5 + .5) * res).to(torch.int64)
            shifted_idx = shifted[..., 0] * res * res + shifted[..., 1] * res + shifted[..., 2]
            assert shifted_idx.max().item() < res * res * res and shifted_idx.min().item() >= 0
            occ_grid[shifted_idx] = 1

        valid_idx = torch.where(occ_grid > 0)[0]
        valid_x = valid_idx // (res * res)
        valid_y = (valid_idx // res) % res
        valid_z = valid_idx % res
        valid_pts = torch.stack([valid_x, valid_y, valid_z], -1)
        valid_pts = (valid_pts / float(res) - .5) * 2.

        def export_occ_grid_ply(    
            save_path=f"./debug/occ_grid_{res}.ply",
            include_input_pts=True,
            max_input_pts=200000
        ):
            import os
            import numpy as np
            import trimesh
            import torch

            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

            occ_cpu = occ_grid.detach().cpu()

            total_voxels = res * res * res

            # -------------------------
            # all voxel indices
            # -------------------------
            all_idx = torch.arange(total_voxels, dtype=torch.int64)

            vx = all_idx // (res * res)
            vy = (all_idx // res) % res
            vz = all_idx % res

            voxel_idx_xyz = torch.stack([vx, vy, vz], dim=-1).float()

            # voxel center 좌표
            # 기존 valid_pts는 idx/res 기준이라 voxel corner에 가깝고,
            # 시각화용 center는 +0.5를 넣는 게 더 정확함.
            voxel_centers = ((voxel_idx_xyz + 0.5) / float(res) - 0.5) * 2.0
            voxel_centers_np = voxel_centers.detach().cpu().numpy().astype(np.float32)

            # -------------------------
            # colors
            # empty = black
            # occupied = red
            # -------------------------
            colors = np.zeros((total_voxels, 3), dtype=np.uint8)

            occ_mask = occ_cpu.numpy() > 0
            colors[occ_mask] = np.array([255, 0, 0], dtype=np.uint8)

            vertices = voxel_centers_np
            vertex_colors = colors

            # -------------------------
            # optionally append original pts
            # input pts = green
            # -------------------------
            if include_input_pts:
                pts_cpu = pts.detach().cpu()
                occ_cpu = occ_grid.detach().cpu()

                n_pts = pts_cpu.shape[0]

                if n_pts > max_input_pts:
                    sample_idx = torch.randperm(
                        n_pts,
                        device='cpu'
                    )[:max_input_pts]
                    pts_vis = pts_cpu[sample_idx]
                else:
                    pts_vis = pts_cpu

                # 무조건 CPU 고정
                pts_vis = pts_vis.detach().cpu()
                occ_cpu = occ_cpu.detach().cpu()

                # -------------------------
                # input pts가 [-1, 1] 안에 있는지 검사
                # -------------------------
                in_bounds = (
                    (pts_vis[:, 0] >= -1.0) & (pts_vis[:, 0] < 1.0) &
                    (pts_vis[:, 1] >= -1.0) & (pts_vis[:, 1] < 1.0) &
                    (pts_vis[:, 2] >= -1.0) & (pts_vis[:, 2] < 1.0)
                ).detach().cpu()

                # -------------------------
                # input pts의 voxel index 계산
                # -------------------------
                pts_voxel = ((pts_vis * 0.5 + 0.5) * res).to(torch.int64)
                pts_voxel = pts_voxel.clamp(0, res - 1).detach().cpu()

                pts_idx = (
                    pts_voxel[:, 0] * res * res +
                    pts_voxel[:, 1] * res +
                    pts_voxel[:, 2]
                ).detach().cpu()

                # -------------------------
                # occupied 여부 판정
                # -------------------------
                pts_is_occupied = torch.zeros(
                    pts_vis.shape[0],
                    dtype=torch.bool,
                    device='cpu'
                )

                pts_is_occupied[in_bounds] = occ_cpu[pts_idx[in_bounds]] > 0

                # -------------------------
                # colors
                # green = input pts and occupied
                # yellow = input pts but NOT occupied
                # -------------------------
                pts_np = pts_vis.detach().cpu().numpy().astype(np.float32)

                pts_colors = np.zeros((pts_np.shape[0], 3), dtype=np.uint8)

                green_mask = pts_is_occupied.numpy()
                yellow_mask = (~pts_is_occupied).numpy()

                pts_colors[green_mask] = np.array([0, 255, 0], dtype=np.uint8)
                pts_colors[yellow_mask] = np.array([255, 255, 0], dtype=np.uint8)

                vertices = np.concatenate([vertices, pts_np], axis=0)
                vertex_colors = np.concatenate([vertex_colors, pts_colors], axis=0)

                num_green = int(pts_is_occupied.sum().item())
                num_yellow = int((~pts_is_occupied).sum().item())

                print(f"[debug] input pts total: {n_pts}, exported: {pts_np.shape[0]}")
                print(f"[debug] green input pts occupied: {num_green}")
                print(f"[debug] yellow input pts NOT occupied: {num_yellow}")

                # -------------------------
                # save ply
                # -------------------------
                pcd = trimesh.PointCloud(vertices, vertex_colors=vertex_colors)
                pcd.export(save_path)

                print(f"[debug] total voxels: {total_voxels}")
                print(f"[debug] occupied voxels: {int(occ_mask.sum())}")
                print(f"[debug] empty voxels: {int((~occ_mask).sum())}")
                print(f"[debug] saved voxel occupancy ply to: {save_path}")


        export_occ_grid_ply()

        return occ_grid, valid_pts

    def state_dict(self):
        ret = dict()
        ret['n_sup_infos'] = len(self.sup_infos)
        for i in range(len(self.sup_infos)):
            ret['sup_info_{}_height'] = self.sup_infos[i].height
            ret['sup_info_{}_width'] = self.sup_infos[i].width
            ret['sup_info_{}'.format(i)] = self.sup_infos[i].state_dict()

        return ret

    def load_state_dict(self, state_dict):
        n_sup_infos = state_dict['n_sup_infos']
        for i in range(n_sup_infos):
            height = state_dict['sup_info_{}_height']
            width  = state_dict['sup_info_{}_width']
            sup_info = PanoSupInfo(pose=torch.eye(4),
                                   mask=torch.ones([height, width, 1]),
                                   color_map=torch.ones([height, width, 3]),
                                   distance_map=torch.ones([height, width, 1]),
                                   normal_map=torch.ones([height, width, 3]))

            sup_info.set_after_reload()
            self.sup_infos.append(sup_info)

        self.all_sup_colors = torch.cat([info.sup_colors for info in self.sup_infos], 0)
        self.all_sup_rays = cat_rays([info.sup_rays for info in self.sup_infos])
        self.all_sup_distances = torch.cat([info.sup_distances for info in self.sup_infos], 0)
        self.all_sup_normals = torch.cat([info.sup_normals for info in self.sup_infos], 0)

    @torch.no_grad()
    def gen_occ_grids_per_view(self, res=256):
        grids = []
        points_per_view = []

        # 3x3x3 voxel 확장
        offsets = torch.tensor(
            [
                [dx, dy, dz]
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for dz in (-1, 0, 1)
            ],
            dtype=torch.long
        )

        for view_idx, sup_info in enumerate(self.sup_infos):
            rays_o, rays_d = sup_info.sup_rays.collapse()
            distances = sup_info.sup_distances.squeeze(-1)

            device = rays_o.device
            offsets_device = offsets.to(device)

            # 해당 view만의 world-space 표면점
            pts = rays_o + rays_d * distances[..., None]

            # AABB 바깥점은 clip하지 않고 제외
            in_bounds = (
                (pts[:, 0] >= -1.0) & (pts[:, 0] < 1.0) &
                (pts[:, 1] >= -1.0) & (pts[:, 1] < 1.0) &
                (pts[:, 2] >= -1.0) & (pts[:, 2] < 1.0)
            )

            pts = pts[in_bounds]
            points_per_view.append(pts)

            voxel_xyz = torch.floor(
                (pts * 0.5 + 0.5) * res
            ).to(torch.long)

            # 각 표면점 주변 3x3x3 voxel 표시
            voxel_xyz = (
                voxel_xyz[:, None, :] +
                offsets_device[None, :, :]
            ).reshape(-1, 3)

            valid = (
                (voxel_xyz[:, 0] >= 0) & (voxel_xyz[:, 0] < res) &
                (voxel_xyz[:, 1] >= 0) & (voxel_xyz[:, 1] < res) &
                (voxel_xyz[:, 2] >= 0) & (voxel_xyz[:, 2] < res)
            )

            voxel_xyz = voxel_xyz[valid]

            flat_idx = (
                voxel_xyz[:, 0] * res * res +
                voxel_xyz[:, 1] * res +
                voxel_xyz[:, 2]
            )

            grid = torch.zeros(
                res ** 3,
                dtype=torch.bool,
                device=device
            )

            grid[flat_idx] = True
            grids.append(grid)

            print(
                f'[view {view_idx}] '
                f'total points={len(in_bounds)}, '
                f'in bounds={in_bounds.sum().item()}, '
                f'occupied voxels={grid.sum().item()}'
            )

        return grids, points_per_view

    @torch.no_grad()
    def export_two_view_occ_comparison(
        self,
        grids,
        resolutions,
        save_path='./debug/occ_comparison.ply'
    ):
        import os
        import numpy as np
        import trimesh
        import torch

        res = resolutions

        if len(grids) != 2:
            raise ValueError(
                f'This visualization expects 2 views, got {len(grids)}'
            )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        g0 = grids[0].detach().cpu()
        g1 = grids[1].detach().cpu()

        only_0 = g0 & (~g1)
        only_1 = g1 & (~g0)
        overlap = g0 & g1
        union = g0 | g1

        occupied_idx = torch.where(union)[0]

        x = occupied_idx // (res * res)
        y = (occupied_idx // res) % res
        z = occupied_idx % res

        voxel_xyz = torch.stack([x, y, z], dim=-1).float()

        # voxel center를 [-1,1] 좌표로 변환
        centers = (
            (voxel_xyz + 0.5) / float(res) - 0.5
        ) * 2.0

        category = torch.zeros_like(occupied_idx, dtype=torch.uint8)

        category[only_0[occupied_idx]] = 1
        category[only_1[occupied_idx]] = 2
        category[overlap[occupied_idx]] = 3

        colors = torch.zeros(
            [len(occupied_idx), 3],
            dtype=torch.uint8
        )

        colors[category == 1] = colors.new_tensor([255, 60, 60]) #view 0 only: red
        colors[category == 2] = colors.new_tensor([60, 100, 255]) #view 1 only: blue
        colors[category == 3] = colors.new_tensor([50, 220, 100]) #overlap: green

        point_cloud = trimesh.PointCloud(
            centers.detach().cpu().numpy().astype(np.float32),
            vertex_colors=colors.detach().cpu().numpy().astype(np.uint8)
        )

        point_cloud.export(save_path)

        intersection_count = overlap.sum().item()
        union_count = union.sum().item()
        iou = intersection_count / max(union_count, 1)

        # breakpoint()
        
        print('view 0 only:', only_0.sum().item())
        print('view 1 only:', only_1.sum().item())
        print('overlap    :', intersection_count)
        print('union      :', union_count)
        print('voxel IoU  :', iou)
        print('saved      :', save_path)

    def load_geometry_ply(self, ply_path):
        import numpy as np
        import trimesh

        geometry = trimesh.load(ply_path, process=False)
        xyz = np.asarray(geometry.vertices, dtype=np.float32).copy()

        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
            raise ValueError(f"Invalid point cloud: {xyz.shape}")

        if not np.isfinite(xyz).all():
            raise ValueError("PLY contains NaN or Inf")

        inside = (np.abs(xyz) <= 1.0).all(axis=1)

        print(f"[PLY] path: {ply_path}")
        print(f"[PLY] points: {len(xyz):,}")
        print("[PLY] XYZ min:", xyz.min(axis=0))
        print("[PLY] XYZ max:", xyz.max(axis=0))
        print(f"[PLY] AABB inside: {inside.mean():.2%}")

        if not inside.all():
            raise ValueError("PLY points exceed [-1, 1]^3")

        # register_sup_info() 이후 호출
        device = self.all_sup_distances.device
        self.ply_points = torch.from_numpy(xyz).to(device)


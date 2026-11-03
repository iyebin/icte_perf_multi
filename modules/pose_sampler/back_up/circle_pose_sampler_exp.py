from ..pose_sampler import PoseSampler

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.camera_utils import *
from scipy.ndimage import minimum_filter1d, gaussian_filter1d


@torch.no_grad()
def _resample_uniformly(pts):
    n = len(pts)
    pts = F.interpolate(pts[None].permute(0, 2, 1), size=n * 128, mode='linear')[0].permute(1, 0)
    cat_pts = torch.cat([pts, pts[:1]], dim=0)
    bias = cat_pts[1:] - cat_pts[:-1]
    bias_len = torch.linalg.norm(bias, 2, -1, keepdim=False)
    bias_len = torch.cumsum(bias_len, dim=0)
    bias_len = bias_len / bias_len[-1]
    idx = torch.searchsorted(bias_len, torch.linspace(0., 1. - 1./n, n))
    return pts[idx]


@torch.no_grad()
def _get_trajectory_normals(pts):
    n_pts = len(pts)
    sigma = float(n_pts) / 32. * 2. + 1.
    ext_pts = torch.cat([pts, pts[:1]], dim=0)
    right_vec = (ext_pts[1:] - ext_pts[:-1])
    right_vec = right_vec / torch.linalg.norm(right_vec, 2, -1, True)
    up_vec = torch.zeros_like(right_vec)
    up_vec[:, 2] = 1
    to_vec = torch.cross(up_vec, right_vec)
    to_vec = to_vec / torch.linalg.norm(to_vec, 2, -1, True)
    to_vec = to_vec.cpu().numpy()
    for i in range(3):
        to_vec[:, i] = gaussian_filter1d(to_vec[:, i], sigma=sigma, mode='wrap')
    to_vec = torch.from_numpy(to_vec).to(pts.device)
    to_vec = to_vec / torch.linalg.norm(to_vec, 2, -1, True)
    return -to_vec


def _process_single_distance_map(distance_map):
    """
    single-view 버전과 동일한 방식으로 단일 distance map에서
    plane_dis(필터 전), filtered_plane_dis, smoothed_plane_dis를 계산.
    카메라 로컬 프레임 기준.
    """
    if torch.is_tensor(distance_map):
        distance_map = distance_map.cpu().numpy()
    distance_map = distance_map.squeeze()

    height, width = distance_map.shape
    pano_coords = img_to_pano_coord(img_coord_from_hw(height, width))

    plane_dis = distance_map * np.cos(pano_coords[:, :, 0].cpu().numpy())
    h_height  = height // 2
    plane_dis = plane_dis[h_height - 10: h_height + 10]
    plane_dis[np.where(plane_dis < 1e-5)] = 1e9
    plane_dis = np.min(plane_dis, axis=0)

    for i in range(1, width):
        if plane_dis[i] > 1e8:
            plane_dis[i] = plane_dis[i - 1]
    for i in range(1, width):
        if plane_dis[width - i - 1] > 1e8:
            plane_dis[width - i - 1] = plane_dis[width - i]

    pool_size          = (width // 16) // 2 * 2 + 1
    filtered_plane_dis = minimum_filter1d(plane_dis, size=pool_size, mode='wrap')
    smooth_size        = (width // 8)  // 2 * 2 + 1
    smoothed_plane_dis = gaussian_filter1d(filtered_plane_dis, sigma=smooth_size, mode='wrap')
    blur_size          = (width // 64) // 2 * 2 + 1
    filtered_plane_dis = gaussian_filter1d(filtered_plane_dis, sigma=blur_size,   mode='wrap')

    return plane_dis, filtered_plane_dis, smoothed_plane_dis, width


class CirclePoseSampler(PoseSampler):
    def __init__(self, distance_maps, poses, traverse_ratios, n_anchors_per_ratio,
                 test_z_min_max=(0., 0.), z_min_ratio=0., z_max_ratio=0., **kwargs):
        super().__init__()

        all_anchor_pts    = []
        all_traverse_pts  = []
        plane_pts_raw     = []
        plane_pts_filter  = []
        plane_pts_smooth  = []

        for distance_map, pose in zip(distance_maps, poses):
            device = distance_map.device
            R_i    = pose[:3, :3].to(device)   # c2w rotation
            t_i    = pose[:3, 3].to(device)    # camera center in world
            cam_z  = t_i[2].item()

            # ── single-view 방식으로 카메라 로컬 plane_dis 계산 ──────────────
            plane_dis_np, filtered_np, smoothed_np, width = \
                _process_single_distance_map(distance_map)

            plane_coords = torch.stack([torch.ones([width]) * .5,
                                        torch.linspace(.5 / width, 1. - .5 / width, width)], -1)
            plane_pts_np = img_coord_to_pano_direction(plane_coords).cpu().numpy()

            # 로컬 벽 위치 → world 변환 (시각화용)
            def _to_world(pts_np, dis_np):
                local = torch.from_numpy(pts_np * dis_np[:, None]).to(device)
                return apply_rot(local, R_i) + t_i

            plane_pts_raw.append(_to_world(plane_pts_np, plane_dis_np))
            plane_pts_filter.append(_to_world(plane_pts_np, filtered_np))
            plane_pts_smooth.append(_to_world(plane_pts_np, smoothed_np))

            filtered_t  = torch.from_numpy(filtered_np).to(device)
            smoothed_t  = torch.from_numpy(smoothed_np).to(device)

            # 수평 단위 방향벡터 (카메라 로컬)
            circle_pts = img_coord_to_pano_direction(plane_coords).to(device)

            # z 오프셋 (해당 카메라 높이 기준)
            mean_radius = float(filtered_t.mean().item())
            z_min = -mean_radius * z_min_ratio
            z_max =  mean_radius * z_max_ratio
            if test_z_min_max != (0., 0.):
                z_min, z_max = test_z_min_max

            # ── 해당 카메라 기준 anchor 생성 ─────────────────────────────────
            for i, traverse_ratio in enumerate(traverse_ratios):
                # 카메라 로컬에서 traverse 궤적 계산
                traverse_pts_local = circle_pts * filtered_t[:, None] * traverse_ratio
                traverse_pts_local = _resample_uniformly(traverse_pts_local)

                # 카메라 로컬 → world 좌표 변환 (c2w: R @ local + t)
                traverse_pts = apply_rot(traverse_pts_local, R_i) + t_i

                n    = n_anchors_per_ratio[i]
                bias = 0. if i % 2 == 0 else .5 / n
                anchor_idx = torch.linspace(.5 / n, 1. - .5 / n, n) + bias
                anchor_idx = (anchor_idx * width).to(torch.long).clip(0, width - 1)

                cur_pts = traverse_pts[anchor_idx].clone()
                for j in range(len(cur_pts)):
                    # z는 world z 방향으로 해당 카메라 높이 기준 오프셋
                    cur_pts[j, 2] = cam_z + (z_min if (i + j) % 2 == 0 else z_max)
                all_anchor_pts.append(cur_pts)

            # ── 해당 카메라 기준 traverse path ───────────────────────────────
            traverse_local = _resample_uniformly(circle_pts * smoothed_t[:, None] * .3)
            all_traverse_pts.append(apply_rot(traverse_local, R_i) + t_i)

        self.anchor_pts = torch.cat(all_anchor_pts, dim=0)

        # 각 카메라의 traverse ring을 이어 붙여 전체 경로 구성
        self.traverse_pts     = torch.cat(all_traverse_pts, dim=0)
        self.traverse_normals = _get_trajectory_normals(self.traverse_pts)

        self.plane_pts_raw    = torch.cat(plane_pts_raw,    dim=0)
        self.plane_pts_filter = torch.cat(plane_pts_filter, dim=0)
        self.plane_pts_smooth = torch.cat(plane_pts_smooth, dim=0)

        self.n_anchors = len(self.anchor_pts)
        self.n_poses   = self.n_anchors

    @torch.no_grad()
    def sample_pose(self, idx):
        pose = torch.eye(4)
        pose[:3, 3] = self.anchor_pts[idx]
        return pose

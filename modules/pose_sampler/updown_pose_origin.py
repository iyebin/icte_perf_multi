from .pose_sampler import PoseSampler

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

class UpDownPoseSampler(PoseSampler):
    def __init__(self, distance_map, traverse_ratios, n_anchors_per_ratio, z_max_ratio, z_min_ratio, test_z_min_max=(0, 0), **kwargs):
        super().__init__()
        
        if torch.is_tensor(distance_map):
            distance_map = distance_map.cpu().numpy()
        distance_map = distance_map.squeeze()

        height, width = distance_map.shape
        pano_coords = img_to_pano_coord(img_coord_from_hw(height, width))

        max_pixel  = int(height * z_max_ratio) #z_max_ratio: 0.5~1 사이
        middle_pixel = int(height * 0.5)
        min_pixel  = int(height * z_min_ratio) #z_min_ratio: 0~0.5 사이이

        plane_dis = distance_map * np.cos(pano_coords[:, :, 0].cpu().numpy())
        
        plane_middle_dis = plane_dis[middle_pixel - 10: middle_pixel + 10]
        plane_up_dis = plane_dis[max_pixel - 10: max_pixel + 10]
        plane_down_dis = plane_dis[min_pixel - 10: min_pixel + 10]

        plane_middle_dis[np.where(plane_middle_dis < 1e-5)] = 1e9
        plane_up_dis[np.where(plane_up_dis < 1e-5)] = 1e9
        plane_down_dis[np.where(plane_down_dis < 1e-5)] = 1e9

        plane_middle_dis = np.min(plane_middle_dis, axis=0)
        plane_up_dis = np.min(plane_up_dis, axis=0)
        plane_down_dis = np.min(plane_down_dis, axis=0)


        for i in range(1, width):
            if plane_middle_dis[i] > 1e8:
                plane_middle_dis[i] = plane_middle_dis[i - 1]

        for i in range(1, width):
            if plane_middle_dis[width - i - 1] > 1e8:
                plane_middle_dis[width - i - 1] = plane_middle_dis[width - i]
        
        for i in range(1, width):
            if plane_up_dis[i] > 1e8:
                plane_up_dis[i] = plane_up_dis[i - 1]

        for i in range(1, width):
            if plane_up_dis[width - i - 1] > 1e8:
                plane_up_dis[width - i - 1] = plane_up_dis[width - i]

        for i in range(1, width):
            if plane_down_dis[i] > 1e8:
                plane_down_dis[i] = plane_down_dis[i - 1]

        for i in range(1, width):
            if plane_down_dis[width - i - 1] > 1e8:
                plane_down_dis[width - i - 1] = plane_down_dis[width - i]


        pool_size = (width // 16) // 2 * 2 + 1
        smooth_size = (width // 8) // 2 * 2 + 1
        blur_size = (width // 64) // 2 * 2 + 1

        filtered_plane_middle_dis = minimum_filter1d(plane_middle_dis, size=pool_size, mode='wrap')
        filtered_plane_up_dis = minimum_filter1d(plane_up_dis, size=pool_size, mode='wrap')
        filtered_plane_down_dis = minimum_filter1d(plane_down_dis, size=pool_size, mode='wrap')

        smoothed_plane_middle_dis = gaussian_filter1d(filtered_plane_middle_dis, sigma=smooth_size, mode='wrap')
        smoothed_plane_up_dis = gaussian_filter1d(filtered_plane_up_dis, sigma=smooth_size, mode='wrap')
        smoothed_plane_down_dis = gaussian_filter1d(filtered_plane_down_dis, sigma=smooth_size, mode='wrap')

        filtered_plane_middle_dis = gaussian_filter1d(filtered_plane_middle_dis, sigma=blur_size, mode='wrap')
        filtered_plane_up_dis = gaussian_filter1d(filtered_plane_up_dis, sigma=blur_size, mode='wrap')
        filtered_plane_down_dis = gaussian_filter1d(filtered_plane_down_dis, sigma=blur_size, mode='wrap')

        # 정규화 픽셀 좌표 생성
        plane_middle_coords = torch.stack([torch.ones([width]) * 0.5, #세로 좌표를 모두 0.5로 고정
                                    torch.linspace(.5 / width, 1. - .5 / width, width)], -1)
        plane_up_coords = torch.stack([torch.ones([width]) * z_max_ratio, #세로 좌표를 모두 z_max_ratio로 고정
                                    torch.linspace(.5 / width, 1. - .5 / width, width)], -1)
        plane_down_coords = torch.stack([torch.ones([width]) * z_min_ratio, #세로 좌표를 모두 z_min_ratio로 고정
                                    torch.linspace(.5 / width, 1. - .5 / width, width)], -1)


        #방향 벡터 생성
        plane_middle_dir = img_coord_to_pano_direction(plane_middle_coords).cpu().numpy()
        plane_up_dir = img_coord_to_pano_direction(plane_up_coords).cpu().numpy()
        plane_down_dir = img_coord_to_pano_direction(plane_down_coords).cpu().numpy()

        #모든 point 얻기 (pts = dir * d)
        self.plane_middle_pts_raw = torch.from_numpy(plane_middle_dir * plane_middle_dis[:, None]).cuda()
        self.plane_middle_pts_filter = torch.from_numpy(plane_middle_dir * filtered_plane_middle_dis[:, None]).cuda()
        self.plane_middle_pts_smooth = torch.from_numpy(plane_middle_dir * smoothed_plane_middle_dis[:, None]).cuda()

        self.plane_up_pts_raw = torch.from_numpy(plane_up_dir * plane_up_dis[:, None]).cuda()
        self.plane_up_pts_filter = torch.from_numpy(plane_up_dir * filtered_plane_up_dis[:, None]).cuda()
        self.plane_up_pts_smooth = torch.from_numpy(plane_up_dir * smoothed_plane_up_dis[:, None]).cuda()

        self.plane_down_pts_raw = torch.from_numpy(plane_down_dir * plane_down_dis[:, None]).cuda()
        self.plane_down_pts_filter = torch.from_numpy(plane_down_dir * filtered_plane_down_dis[:, None]).cuda()
        self.plane_down_pts_smooth = torch.from_numpy(plane_down_dir * smoothed_plane_down_dis[:, None]).cuda()

        #filtered, smoothed torch에서 npy로
        filtered_plane_middle_dis = torch.from_numpy(filtered_plane_middle_dis).cuda()
        smoothed_plane_middle_dis = torch.from_numpy(smoothed_plane_middle_dis).cuda()

        filtered_plane_up_dis = torch.from_numpy(filtered_plane_up_dis).cuda()
        smoothed_plane_up_dis = torch.from_numpy(smoothed_plane_up_dis).cuda()

        filtered_plane_down_dis = torch.from_numpy(filtered_plane_down_dis).cuda()
        smoothed_plane_down_dis = torch.from_numpy(smoothed_plane_down_dis).cuda()

        #circle pts
        circle_middle_pts = img_coord_to_pano_direction(plane_middle_coords)
        circle_up_pts = img_coord_to_pano_direction(plane_up_coords)
        circle_down_pts = img_coord_to_pano_direction(plane_down_coords)


        anchor_middle_pts = []
        anchor_up_pts = []
        anchor_down_pts = []
        test_z_min, test_z_max = test_z_min_max

        for i, traverse_ratio in enumerate(traverse_ratios):
            traverse_middle_pts = circle_middle_pts * filtered_plane_middle_dis[:, None] * traverse_ratio
            traverse_up_pts = circle_up_pts * filtered_plane_up_dis[:, None] * traverse_ratio
            traverse_down_pts = circle_down_pts * filtered_plane_down_dis[:, None] * traverse_ratio

            traverse_middle_pts = _resample_uniformly(traverse_middle_pts)
            traverse_up_pts = _resample_uniformly(traverse_up_pts)
            traverse_down_pts = _resample_uniformly(traverse_down_pts)

            n = n_anchors_per_ratio[i]
            bias = 0. if i % 2 == 0 else .5 / n
            anchor_idx = torch.linspace(.5 / n, 1. - .5 / n, n) + bias
            anchor_idx = (anchor_idx * width).to(torch.long).clip(0, width - 1)

            cur_middle_pts = traverse_middle_pts[anchor_idx].clone()
            cur_up_pts = traverse_up_pts[anchor_idx].clone()
            cur_down_pts = traverse_down_pts[anchor_idx].clone()

            for j in range(len(cur_middle_pts)):
                cur_middle_pts[j, 2] = test_z_min if (i + j) % 2 == 0 else test_z_max
            for j in range(len(cur_up_pts)):
                cur_up_pts[j, 2] = test_z_min if (i + j) % 2 == 0 else test_z_max
            for j in range(len(cur_down_pts)):
                cur_down_pts[j, 2] = test_z_min if (i + j) % 2 == 0 else test_z_max
            
            anchor_middle_pts.append(cur_middle_pts)
            anchor_up_pts.append(cur_up_pts)
            anchor_down_pts.append(cur_down_pts)

            traverse_middle_pts[..., 2] += (test_z_min + test_z_max) * .5
            traverse_up_pts[..., 2] += (test_z_min + test_z_max) * .5
            traverse_down_pts[..., 2] += (test_z_min + test_z_max) * .5
        
        self.anchor_middle_pts = torch.cat(anchor_middle_pts, dim=0)
        self.anchor_up_pts = torch.cat(anchor_up_pts, dim=0)
        self.anchor_down_pts = torch.cat(anchor_down_pts, dim=0)

        self.anchor_pts = torch.cat([self.anchor_middle_pts,
                                     self.anchor_up_pts,
                                     self.anchor_down_pts], dim=0)

        self.traverse_middle_pts = _resample_uniformly(circle_middle_pts * smoothed_plane_middle_dis[:, None] * .3)
        self.traverse_up_pts = _resample_uniformly(circle_up_pts * smoothed_plane_up_dis[:, None] * .3)
        self.traverse_down_pts = _resample_uniformly(circle_down_pts * smoothed_plane_down_dis[:, None] * .3)

        self.n_anchors = len(self.anchor_pts)
        self.n_poses = self.n_anchors

    @torch.no_grad()
    def sample_pose(self, idx):
        pose = torch.eye(4)
        pose[:3, 3] = self.anchor_pts[idx]
        return pose
        

    
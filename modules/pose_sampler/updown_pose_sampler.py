from .pose_sampler import PoseSampler
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.camera_utils import *
from scipy.ndimage import minimum_filter1d, gaussian_filter1d

@torch.no_grad()
def _resample_uniformly(pts):
    # pts를 균일하게 위치 맞추는 function
    n = len(pts)
    pts = F.interpolate(pts[None].permute(0, 2, 1), size=n * 128, mode='linear')[0].permute(1, 0)
    cat_pts = torch.cat([pts, pts[:1]], dim=0)
    bias = cat_pts[1:] - cat_pts[:-1]
    bias_len = torch.linalg.norm(bias, 2, -1, keepdim=False)
    bias_len = torch.cumsum(bias_len, dim=0)
    bias_len = bias_len / bias_len[-1]
    idx = torch.searchsorted(bias_len, torch.linspace(0., 1. - 1./n, n))
    return pts[idx]

def _outlier_interpolation(arr):
    # 1e8 이상인 값을 이전값으로 채우는 function
    width = len(arr)

    for i in range(1, width):
        if arr[i] > 1e8:
            arr[i] = arr[i-1]
    
    for i in range(1, width):
        if arr[width - i - 1] > 1e8:
            arr[width - i - 1] = arr[width - i]

    return arr

def _return_minimum_distance(plane_dis_full, pixel_x, half_band=10):
    # plane_dis_full에서 10px 만큼 pixel row를 중심으로 최소 거리 반환 function
    band = plane_dis_full[pixel_x - half_band: pixel_x + half_band].copy()
    band[band < 1e-5] = 1e9
    dis = np.min(band, axis=0)
    dis = _outlier_interpolation(dis)

    return dis

def _compute_filter_smooth(raw_dis, width):
    # minimum filter + gaussian smooth + blur 적용 function
    pool_size = (width // 16) // 2 * 2 + 1
    smooth_size = (width // 8) // 2 * 2 + 1
    blur_size = (width // 64) // 2 * 2 + 1

    filtered = minimum_filter1d(raw_dis, size=pool_size, mode='wrap')
    smoothed = gaussian_filter1d(filtered, sigma=smooth_size, mode='wrap')
    # 여기서 filtered를 다시 계산하는 것이 의도된 것인지 확인 필요

    filtered = gaussian_filter1d(filtered, sigma=blur_size, mode='wrap')

    return filtered, smoothed


def _save_tensor_txt(tensor, filename):
    arr = tensor.detach().cpu().numpy()
    np.savetxt(filename, arr, fmt="%.6f")


class UpDownPoseSampler(PoseSampler):
    # middle, up, down 3개의 수평 곡선을 따라 pose를 생성하는 sampler
    def __init__(self, distance_map, n_rings=3, traverse_ratios=None, n_anchors_per_ratio=None, 
                 z_max_ratio=0.7, z_min_ratio=0.3, z_max=0.0, z_min=0.0, **kwargs):
        super().__init__()
        
        # 클래스 상수들 정의
        self.middle = 0
        self.up = 1
        self.down = 2
        self.circle_middle = 0
        self.circle_up = 1
        self.circle_down = 2
        
        # 기본값 설정
        if traverse_ratios is None:
            traverse_ratios = [0.8, 0.9, 1.0]
        if n_anchors_per_ratio is None:
            n_anchors_per_ratio = [8, 12, 16]
            
        if torch.is_tensor(distance_map):
            distance_map = distance_map.cpu().numpy()
        distance_map = distance_map.squeeze()

        height, width = distance_map.shape
        pano_coords = img_to_pano_coord(img_coord_from_hw(height, width))

        full_plane_dis = distance_map * np.cos(pano_coords[:, :, 0].cpu().numpy())

        all_pixel_line_position = {
            self.middle: int(height * 0.5),
            self.up: int(height * z_max_ratio),
            self.down: int(height * z_min_ratio)
        }

        normed_coords = torch.linspace(.5 / width, 1. - .5 / width, width)

        all_data = {}
        # id = middle, up, down
        for id, pixel_x in all_pixel_line_position.items():
            raw_dis = _return_minimum_distance(full_plane_dis, pixel_x)
            filtered_dis, smoothed_dis = _compute_filter_smooth(raw_dis, width)

            v = all_pixel_line_position[id] / height
            plane_coords = torch.stack(
                [torch.ones(width) * v, normed_coords], dim=-1
            )

            circle_pts = img_coord_to_pano_direction(plane_coords)
            plane_pts = circle_pts.cpu().numpy()

            all_data[id] = {
                'raw_dis': raw_dis,
                'filtered_dis': filtered_dis,
                'smoothed_dis': smoothed_dis,
                'circle_pts': circle_pts,
                'pts_raw': torch.from_numpy(plane_pts * raw_dis[:, None]).cuda(),
                'pts_filter': torch.from_numpy(plane_pts * filtered_dis[:, None]).cuda(),
                'pts_smooth': torch.from_numpy(plane_pts * smoothed_dis[:, None]).cuda(),
            }

        for id, name in [(0, 'middle'), (1, 'up'), (2, 'down')]:
            data = all_data[id]
            setattr(self, f'plane_{name}_pts_raw', data['pts_raw'])
            setattr(self, f'plane_{name}_pts_filter', data['pts_filter'])
            setattr(self, f'plane_{name}_pts_smooth', data['pts_smooth'])

        for id in all_data:
            data = all_data[id]
            data['filtered_dis_t'] = torch.from_numpy(data['filtered_dis']).cuda()
            data['smoothed_dis_t'] = torch.from_numpy(data['smoothed_dis']).cuda()

        anchor_pts_per_circle = {
            self.circle_middle: [], 
            self.circle_up: [], 
            self.circle_down: []
        }
        anchor_pts_direction = []
        for i, traverse_ratio in enumerate(traverse_ratios):
            n = n_anchors_per_ratio[i]
            bias = 0. if i % 2 == 0 else .5 / n
            anchor_idx = torch.linspace(.5 / n, 1. - .5 / n, n) + bias
            anchor_idx = (anchor_idx * width).to(torch.long).clip(0, width - 1)

            # z_pattern = torch.tensor(
            #     [z_min if (i+j) % 2 == 0 else z_max for j in range(n)]
            # ).cuda()

            for id in [self.circle_middle, self.circle_up, self.circle_down]:
                data = all_data[id]
                traverse_pts = data['circle_pts'] * data['filtered_dis_t'][:, None] * traverse_ratio
                anchor_coord = traverse_pts / data['filtered_dis_t'][:, None] / traverse_ratio
                anchor_pts_direction.append(anchor_coord)
                traverse_pts = _resample_uniformly(traverse_pts)

                cur_pts = traverse_pts[anchor_idx].clone()
                # cur_pts[:, 2] = z_pattern

                anchor_pts_per_circle[id].append(cur_pts) 
        self.anchor_pts_direction = torch.stack(anchor_pts_direction)
            
        anchors_middle = torch.cat(anchor_pts_per_circle[self.circle_middle], dim=0)
        anchors_up = torch.cat(anchor_pts_per_circle[self.circle_up], dim=0)
        anchors_down = torch.cat(anchor_pts_per_circle[self.circle_down], dim=0)

        # iterative: [mid0, up0, down0, mid1, up1, down1, ...] -> (N*3, 3)
        n_anchors_percircle = len(anchors_middle)
        self.anchor_pts = torch.stack(
            [anchors_middle, anchors_up, anchors_down], dim=1
        ).reshape(-1, 3)

        traverse_list = []
        for id in [self.circle_middle, self.circle_up, self.circle_down]:
            data = all_data[id]
            tpts = _resample_uniformly(
                data['circle_pts'] * data['smoothed_dis_t'][:, None] * 0.3  # 0.3은 traverse 비율
            )
            traverse_list.append(tpts)
        
        self.traverse_pts = torch.stack(traverse_list, dim=0)
        self.n_anchors = len(self.anchor_pts)
        self.n_poses = self.n_anchors


        # debug: save anchor positions
        os.makedirs("debug_pose_sampler", exist_ok=True)
        _save_tensor_txt(self.anchor_pts, "debug_pose_sampler/anchor_pts.txt")
        _save_tensor_txt(anchors_middle, "debug_pose_sampler/middle.txt")
        _save_tensor_txt(anchors_up, "debug_pose_sampler/up.txt")
        _save_tensor_txt(anchors_down, "debug_pose_sampler/down.txt")
        

        for i, arr in enumerate(self.anchor_pts_direction):
            _save_tensor_txt(arr, f"debug_pose_sampler/anchor_dir_{i}.txt")

        np.save("debug_pose_sampler/anchor_pts.npy",
        self.anchor_pts.detach().cpu().numpy())

        _save_tensor_txt(circle_pts, "debug_pose_sampler/circle_dir.txt")
        np.savetxt("debug_pose_sampler/raw_distance.txt", raw_dis)
        _save_tensor_txt(traverse_pts, "debug_pose_sampler/traverse_pts.txt")
        _save_tensor_txt(torch.tensor([self.n_poses]), "debug_pose_sampler/n_poses(n_anchors).txt")

    @torch.no_grad()
    def sample_pose(self, idx):
        pose = torch.eye(4)
        pose[:3, 3] = self.anchor_pts[idx]
        return pose


    
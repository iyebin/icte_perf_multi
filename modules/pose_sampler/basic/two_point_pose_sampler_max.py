from .pose_sampler import PoseSampler

import torch
import torch.nn.functional as F
import numpy as np

from utils.camera_utils import *
from scipy.ndimage import minimum_filter1d, gaussian_filter1d


class TwoPointPoseSamplerMax(PoseSampler):
    """두 개의 고정 포즈만 샘플: base_point, base_point + (0, 0, z_offset)."""

    def __init__(self, distance_map, base_point=[0., 0., 0.], z_offset_ratio=0.8):
        super().__init__()
        
        if torch.is_tensor(distance_map):
            distance_map = distance_map.cpu().numpy()
        distance_map = distance_map.squeeze()

        height, width = distance_map.shape
        pano_coords = img_to_pano_coord(img_coord_from_hw(height, width))

        max_z_distance = np.max(distance_map * np.sin(pano_coords[:, :, 1].cpu().numpy()))
        base_point = torch.as_tensor(base_point, dtype=torch.float32)
        z_offset = float(z_offset_ratio * max_z_distance)

        self.anchor_pts = torch.stack([
            base_point,
            base_point + torch.tensor([0., 0., z_offset], dtype=base_point.dtype)
        ])

        self.n_poses = 2
        self.n_anchors = 2  # core_exp_runner 호환

    @torch.no_grad()
    def sample_pose(self, idx):
        pose = torch.eye(4)
        pose[:3, 3] = self.anchor_pts[idx]
        return pose


from .pose_sampler import PoseSampler

import torch
import torch.nn.functional as F
import numpy as np

from utils.camera_utils import *
from scipy.ndimage import minimum_filter1d, gaussian_filter1d


class ThreePoseSampler(PoseSampler):
    """세 개의 고정 포즈만 샘플: base_point, base_point + (0, 0, z_offset)."""

    def __init__(self, base_point=[0., 0., 0.], z_up_offset=0.2, z_down_offset=-0.2):
        super().__init__()
        base_point = torch.as_tensor(base_point, dtype=torch.float32)
        z_up_offset = float(z_up_offset)
        z_down_offset = float(z_down_offset)

        self.anchor_pts = torch.stack([
            base_point + torch.tensor([0., 0., z_down_offset], dtype=base_point.dtype),
            base_point,
            base_point + torch.tensor([0., 0., z_up_offset], dtype=base_point.dtype),
            
        ])

        self.n_poses = 3
        self.n_anchors = 3  # core_exp_runner 호환

    @torch.no_grad()
    def sample_pose(self, idx):
        pose = torch.eye(4)
        pose[:3, 3] = self.anchor_pts[idx]
        return pose



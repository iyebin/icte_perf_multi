from .pose_sampler import PoseSampler

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from utils.camera_utils import *
from scipy.ndimage import minimum_filter1d, gaussian_filter1d

class MiddlePanoramaSampler(PoseSampler):
    """두 pose 사이를 따라 이동하면서 각 지점에서 horizontal sampling"""

    def __init__(self, base_point=[0., 0., 0.], another=[-0.05, 0.0, 0.0]):
        super().__init__()
        base_point = torch.as_tensor(base_point, dtype=torch.float32)
        # z_up_offset = float(z_up_offset)
        # z_down_offset = float(z_down_offset)

        self.anchor_pts = torch.stack([
            base_point,
            base_point + torch.tensor(another, dtype=base_point.dtype),
            
        ])

        self.n_poses = 2
        self.n_anchors = 2  # core_exp_runner 호환
        
    @torch.no_grad()
    def sample_pose(self, idx):
        pose = torch.eye(4)
        pose[:3, 3] = self.anchor_pts[idx]
        return pose

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

    def __init__(self, poses):
        """
        poses: (2, 4, 4)
        n_interp: pose1→pose2 사이 몇 개 지점
        radius: 각 지점에서 퍼지는 반경
        n_horizontal: 각 지점당 horizontal 샘플 개수
        """
        super().__init__()
        n_poses = len(poses)
        poses = torch.stack(poses)
        # poses = torch.inverse(poses) #camera to world transform
        
        p0 = poses[0][:3, 3]
        p1 = poses[n_poses-1][:3, 3]

        p0 = torch.as_tensor(p0, dtype=torch.float32)
        p1 = torch.as_tensor(p1, dtype=torch.float32)

        self.anchor_pts = torch.stack([p0, p1])

        self.n_poses = len(self.anchor_pts)
        self.n_anchors = self.n_poses
        
    @torch.no_grad()
    def sample_pose(self, idx):
        pose = torch.eye(4)
        pose[:3, 3] = self.anchor_pts[idx]
        return pose
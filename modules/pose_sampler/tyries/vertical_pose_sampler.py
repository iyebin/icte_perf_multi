from .pose_sampler import PoseSampler

import torch
import torch.nn.functional as F
import numpy as np

from utils.camera_utils import *
from scipy.ndimage import minimum_filter1d, gaussian_filter1d


@torch.no_grad()
def _resample_uniformly(pts):
    n = len(pts)
    pts = F.interpolate(
        pts[None].permute(0, 2, 1),
        size=n * 128,
        mode='linear'
    )[0].permute(1, 0)

    cat_pts = torch.cat([pts, pts[:1]], dim=0)
    bias = cat_pts[1:] - cat_pts[:-1]
    bias_len = torch.linalg.norm(bias, 2, -1)

    bias_len = torch.cumsum(bias_len, dim=0)
    bias_len = bias_len / bias_len[-1]

    idx = torch.searchsorted(
        bias_len,
        torch.linspace(0., 1. - 1./n, n)
    )

    return pts[idx]


@torch.no_grad()
def _get_trajectory_normals(pts):
    n_pts = len(pts)
    sigma = float(n_pts) / 32. * 2. + 1.

    ext_pts = torch.cat([pts, pts[:1]], dim=0)

    right_vec = ext_pts[1:] - ext_pts[:-1]
    right_vec = right_vec / torch.linalg.norm(right_vec, 2, -1, True)

    up_vec = torch.zeros_like(right_vec)
    up_vec[:, 2] = 1

    to_vec = torch.cross(up_vec, right_vec)
    to_vec = to_vec / torch.linalg.norm(to_vec, 2, -1, True)

    to_vec = to_vec.cpu().numpy()

    for i in range(3):
        to_vec[:, i] = gaussian_filter1d(
            to_vec[:, i],
            sigma=sigma,
            mode='wrap'
        )

    to_vec = torch.from_numpy(to_vec).to(pts.device)
    to_vec = to_vec / torch.linalg.norm(to_vec, 2, -1, True)

    return -to_vec


class VerticalPoseSampler(PoseSampler):

    def __init__(self, distance_map, n_anchors_per_ratio, test_z_min_max=(0., 0.), **kwargs):

        super().__init__()

        if torch.is_tensor(distance_map):
            distance_map = distance_map.cpu().numpy()

        distance_map = distance_map.squeeze()

        height, width = distance_map.shape

        pano_coords = img_to_pano_coord(img_coord_from_hw(height, width))

        plane_dis = distance_map * np.cos(pano_coords[:, :, 0].cpu().numpy())

        h_height = height // 2
        plane_dis = plane_dis[h_height - 10: h_height + 10]

        plane_dis[np.where(plane_dis < 1e-5)] = 1e9
        plane_dis = np.min(plane_dis, axis=0)

        for i in range(1, width):
            if plane_dis[i] > 1e8:
                plane_dis[i] = plane_dis[i - 1]

        for i in range(1, width):
            if plane_dis[width - i - 1] > 1e8:
                plane_dis[width - i - 1] = plane_dis[width - i]

        pool_size = (width // 16) // 2 * 2 + 1
        filtered_plane_dis = minimum_filter1d(plane_dis, size=pool_size, mode='wrap')

        smooth_size = (width // 8) // 2 * 2 + 1
        smoothed_plane_dis = gaussian_filter1d(filtered_plane_dis, sigma=smooth_size, mode='wrap')

        blur_size = (width // 64) // 2 * 2 + 1
        filtered_plane_dis = gaussian_filter1d(filtered_plane_dis, sigma=blur_size, mode='wrap')
        
        
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        filtered_plane_dis = torch.from_numpy(filtered_plane_dis).cuda()
        smoothed_plane_dis = torch.from_numpy(smoothed_plane_dis).cuda()
        # filtered_plane_dis = torch.from_numpy(filtered_plane_dis).to(device)
        # smoothed_plane_dis = torch.from_numpy(smoothed_plane_dis).to(device)

        # --------------------------
        # Vertical trajectory 생성
        # --------------------------

        test_z_min, test_z_max = test_z_min_max

        center_x = 0.0
        center_y = filtered_plane_dis.mean().item()

        anchor_pts = []

        for i, n in enumerate(n_anchors_per_ratio):

            z_vals = torch.linspace(test_z_min, test_z_max, n)

            cur_pts = torch.zeros(n, 3).cuda()
            # cur_pts = torch.zeros(n, 3).to(device)

            cur_pts[:, 0] = center_x
            cur_pts[:, 1] = center_y
            cur_pts[:, 2] = z_vals

            anchor_pts.append(cur_pts)

        self.anchor_pts = torch.cat(anchor_pts, dim=0)

        # trajectory for normal estimation
        z_line = torch.linspace(test_z_min, test_z_max, width)

        self.traverse_pts = torch.stack([
            torch.tensor([center_x, center_y, z]).cuda()
            # torch.tensor([center_x, center_y, z]).to(device)
            for z in z_line
        ])

        self.traverse_normals = _get_trajectory_normals(self.traverse_pts)

        self.n_anchors = len(self.anchor_pts)
        self.n_poses = self.n_anchors


    @torch.no_grad()
    def sample_pose(self, idx):

        pose = torch.eye(4, device=self.anchor_pts.device)

        pos = self.anchor_pts[idx]

        target = torch.tensor([0.,0.,0.], device=pos.device)

        forward = target - pos
        forward = forward / torch.norm(forward)

        up = torch.tensor([0.,0.,1.], device=pos.device)

        if torch.abs(torch.dot(forward, up)) > 0.99:
            up = torch.tensor([0.,1.,0.], device=pos.device)

        right = torch.cross(up, forward)
        right = right / torch.norm(right)

        up = torch.cross(forward, right)

        pose[:3,0] = right
        pose[:3,1] = up
        pose[:3,2] = forward
        pose[:3,3] = pos

        return pose


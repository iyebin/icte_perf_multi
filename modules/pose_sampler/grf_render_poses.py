import torch
import numpy as np

from .pose_sampler import PoseSampler
from utils.camera_utils import *

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

def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec2, vec1_avg))
    vec1 = normalize(np.cross(vec0, vec2))
    m = np.stack([-vec0, vec1, vec2, pos], 1)
    return m


def pose_inverse(pose):
    R = pose[:, :3].T
    t = - R @ pose[:, 3:]
    return np.concatenate([R, t], -1)


def transform_points_Rt(pts, R, t):
    t = t.flatten()
    return pts @ R.T + t[None, :]


def render_path_spiral(c2w, up, rads, focal, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])

    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([
                -np.sin(theta),
                np.cos(theta),
                -np.sin(theta * zrate),
                1.
            ]) * rads
        )
        z = normalize(
            np.dot(c2w[:3, :4], np.array([0, 0, focal, 1.])) - c
        )
        render_poses.append(
            np.concatenate([viewmatrix(z, up, c)], 1)
        )

    return render_poses


class GRFCirclePoseSampler(PoseSampler):
    def __init__(self, distance_maps, poses, N=60):
        super().__init__()

        # -----------------------------
        # 1. numpy 변환
        # -----------------------------
        poses_np = []
        for p in poses:
            if torch.is_tensor(p):
                poses_np.append(p.cpu().numpy())
            else:
                poses_np.append(p)
        poses_np = np.asarray(poses_np)

        # -----------------------------
        # 2. world -> camera (inverse)
        # -----------------------------
        poses_inv = [pose_inverse(p) for p in poses_np]
        poses_inv = np.asarray(poses_inv)

        cam_pts  = poses_inv[:, :, 3]
        cam_rots = poses_inv[:, :, :3]

        down   = cam_rots[:, :, 1]
        lookat = cam_rots[:, :, 2]

        # -----------------------------
        # 3. 평균 카메라 계산
        # -----------------------------
        avg_cam_pt  = (np.max(cam_pts,0) + np.min(cam_pts,0)) / 2.
        avg_down    = normalize(np.mean(down,0))
        avg_lookat  = normalize(np.mean(lookat,0))

        avg_pose_inv = viewmatrix(avg_lookat, avg_down, avg_cam_pt)
        avg_pose     = pose_inverse(avg_pose_inv)

        # -----------------------------
        # 4. 카메라 분포 범위
        # -----------------------------
        cam_pts_in_avg = transform_points_Rt(
            cam_pts,
            avg_pose[:, :3],
            avg_pose[:, 3]
        )

        range_in_avg_pose = np.percentile(
            np.abs(cam_pts_in_avg), 90, axis=0
        )

        # z 방향 흔들림 제한
        range_in_avg_pose[2] = range_in_avg_pose[2] * 0.2

        # shrink
        range_in_avg_pose *= 0.8

        # -----------------------------
        # 5. depth 기반 near / far
        # -----------------------------
        depth_ranges = np.array([
            [
                np.percentile(d.cpu().numpy(), 5),
                np.percentile(d.cpu().numpy(), 95)
            ]
            for d in distance_maps
        ])

        near = np.mean(depth_ranges[:, 0])
        far  = np.mean(depth_ranges[:, 1])

        dt = 0.75
        mean_dz = 1. / (((1. - dt) / near + dt / far))

        # -----------------------------
        # 6. spiral path 생성
        # -----------------------------
        render_poses = render_path_spiral(
            avg_pose_inv,
            avg_down,
            range_in_avg_pose,
            mean_dz,
            zrate=0.,
            rots=1,
            N=N
        )

        render_poses = [pose_inverse(p) for p in render_poses]
        render_poses = np.asarray(render_poses)

        # -----------------------------
        # 7. anchor pts 저장
        # -----------------------------
        self.render_poses = torch.from_numpy(render_poses).float()
        self.anchor_pts   = self.render_poses[:, :3, 3]

        self.n_poses = len(self.anchor_pts)
        self.n_anchors = len(self.anchor_pts)

    @torch.no_grad()
    def sample_pose(self, idx):
        pose = torch.eye(4)
        pose[:3, :4] = self.render_poses[idx]
        return pose

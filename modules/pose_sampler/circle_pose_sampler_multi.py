from .pose_sampler import PoseSampler
import matplotlib.pyplot as plt

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


class CirclePoseSampler(PoseSampler):
    def __init__(self, distance_map, poses, traverse_ratios, n_anchors_per_ratio, test_z_min_max=(0., 0.), **kwargs):
        super().__init__()
        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        map_styles = [
            ('#c9c9c9', '#1f77b4', 'o'),    # map0: raw 회색 / filtered 파랑 / anchor 점(●)
            ('#c9c9c9', '#ff7f0e', '^'),    # map1: raw 회색 / filtered 주황 / anchor 세모(▲)
        ]

        all_x, all_y = [], []

        for n in range(0, 2):
            ax = axes[n]
            distance = distance_map[n]
            if torch.is_tensor(distance):
                distance = distance.cpu().numpy()
            distance = distance.squeeze()

            t = poses[n][:3, 3].cpu().numpy()

            height, width = distance.shape

            pano_coords = img_to_pano_coord(img_coord_from_hw(height, width))

            plane_dis = distance * np.cos(pano_coords[:, :, 0].cpu().numpy())
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

            # --- CUDA로 바꾸기 전에 시각화용 numpy 복사본 보관 ---
            plane_dis_np = plane_dis.copy()
            filtered_np  = filtered_plane_dis.copy()
            smoothed_np = smoothed_plane_dis.copy()

            plane_coords = torch.stack([torch.ones([width]) * .5,
                                        torch.linspace(.5 / width, 1. - .5 / width, width)], -1)
            plane_pts = img_coord_to_pano_direction(plane_coords).cpu().numpy()
            self.plane_pts_raw    = torch.from_numpy(plane_pts * plane_dis[:, None] + t).cuda()
            self.plane_pts_filter = torch.from_numpy(plane_pts * filtered_np[:, None]+ t).cuda()
            self.plane_pts_smooth = torch.from_numpy(plane_pts * smoothed_np[:, None] + t).cuda()
            filtered_plane_dis = torch.from_numpy(filtered_plane_dis).cuda()
            smoothed_plane_dis = torch.from_numpy(smoothed_plane_dis).cuda()          

            # Get anchors
            n_anchors_per_ratio = n_anchors_per_ratio
            plane_coords = torch.stack([torch.ones([width]) * .5,
                                        torch.linspace(.5 / width, 1. - .5 / width, width)], -1)

            circle_pts = img_coord_to_pano_direction(plane_coords)

            anchor_pts = []
            test_z_min, test_z_max = test_z_min_max
            for i, traverse_ratio in enumerate(traverse_ratios):
                traverse_pts = circle_pts * filtered_plane_dis[:, None] * traverse_ratio
                #t 더함
                traverse_pts = traverse_pts + torch.from_numpy(t).cuda().float()
                traverse_pts = _resample_uniformly(traverse_pts)
                k = n_anchors_per_ratio[i]
                bias = 0. if i % 2 == 0 else .5 / k
                anchor_idx = torch.linspace(.5 / k, 1. - .5 / k, k) + bias
                anchor_idx = (anchor_idx * width).to(torch.long).clip(0, width - 1)
                cur_pts = traverse_pts[anchor_idx].clone()
                for j in range(len(cur_pts)):
                    cur_pts[j, 2] = test_z_min if (i + j) % 2 == 0 else test_z_max
                anchor_pts.append(cur_pts)

                traverse_pts[..., 2] += (test_z_min + test_z_max) * .5

            # ratio별 그룹 보관 (선 잇기용) — cat 전에
            anchor_groups = [g.detach().cpu().numpy() for g in anchor_pts]

            self.anchor_pts = torch.cat(anchor_pts, dim=0)
            self.traverse_pts = _resample_uniformly(circle_pts * smoothed_plane_dis[:, None] * .3 + torch.from_numpy(t).cuda().float())
            self.traverse_normals = _get_trajectory_normals(self.traverse_pts)
            self.n_anchors = len(self.anchor_pts)
            self.n_poses = self.n_anchors

            # === 겹쳐 그리기 ===
            # azimuth = pano_coords[0, :, 1].cpu().numpy()
            # raw_c, filt_c, marker = map_styles[n]        # ← 세 번째를 marker로

            # for dis, color, tag, sz in [(plane_dis_np, raw_c,  f'map{n} raw',      3),
            #                             (filtered_np,  filt_c, f'map{n} filtered', 8)]:
            #     valid = dis < 1e8
            #     x = dis[valid] * np.cos(azimuth[valid]) + t[0]
            #     y = dis[valid] * np.sin(azimuth[valid]) + t[1]
            #     axes.scatter(x, y, s=sz, c=color, label=tag)
            #     all_x.append(x); all_y.append(y)

             # === 각 서브플롯에 그리기 ===
            azimuth = pano_coords[0, :, 1].cpu().numpy()
            raw_c, filt_c, marker = map_styles[n]

            for dis, color, tag, sz in [(plane_dis_np, raw_c,  f'map{n} raw',      3),
                                        (filtered_np,  filt_c, f'map{n} filtered', 8)]:
                valid = dis < 1e8
                x = dis[valid] * np.cos(azimuth[valid]) + t[0]
                y = dis[valid] * np.sin(azimuth[valid]) + t[1]
                ax.scatter(x, y, s=sz, c=color, label=tag)        # ← ax
                all_x.append(x); all_y.append(y)

            # === 앵커: ratio별로 선으로 잇기 ===
            # linestyles = ['-', '--', ':']              # ratio 0.2 / 0.4 / 0.6 구분

            # for gi, g in enumerate(anchor_groups):
            #     gx, gy = g[:, 0], g[:, 1]
            #     # 링이므로 첫 점으로 닫아줌
            #     gx_c = np.append(gx, gx[0])
            #     gy_c = np.append(gy, gy[0])
            #     ls = linestyles[gi % len(linestyles)]
            #     axes.plot(gx_c, gy_c, ls=ls, c=filt_c, lw=1.3,
            #               label=f'map{n} ratio={traverse_ratios[gi]}')
            #     axes.scatter(gx, gy, s=60, marker=marker, c=filt_c,
            #                  edgecolors='black', linewidths=0.8, zorder=6)
            #     all_x.append(gx); all_y.append(gy)

             # === 앵커: ratio별로 선으로 잇기 ===
            linestyles = ['-', '--', ':']
            for gi, g in enumerate(anchor_groups):
                gx, gy = g[:, 0], g[:, 1]
                gx_c = np.append(gx, gx[0])
                gy_c = np.append(gy, gy[0])
                ls = linestyles[gi % len(linestyles)]
                ax.plot(gx_c, gy_c, ls=ls, c=filt_c, lw=1.3,      # ← ax
                        label=f'map{n} ratio={traverse_ratios[gi]}')
                ax.scatter(gx, gy, s=60, marker=marker, c=filt_c, # ← ax
                        edgecolors='black', linewidths=0.8, zorder=6)
                all_x.append(gx); all_y.append(gy)

            ax.scatter(t[0], t[1], c='red', marker='*', s=150, label='camera', zorder=7)  # ← ax, label 항상


            # anchor_np = self.anchor_pts.detach().cpu().numpy()
            # axes.scatter(anchor_np[:, 0], anchor_np[:, 1],
            #              s=70, marker=marker, c=filt_c,          # ← 색은 filtered 색, 마커는 맵별
            #              edgecolors='black', linewidths=0.8,
            #              label=f'map{n} anchors ({len(anchor_np)})', zorder=6)
            # all_x.append(anchor_np[:, 0]); all_y.append(anchor_np[:, 1])

            # axes.scatter(t[0], t[1], c='red', marker='*', s=150, label='camera' if n == 0 else None, zorder=7)
            ax.scatter(t[0], t[1], c='red', marker='*', s=150, label='camera' if n == 0 else None, zorder=7)

        # all_x = np.concatenate(all_x); all_y = np.concatenate(all_y)
        # m = max(np.abs(all_x).max(), np.abs(all_y).max()) * 1.05
        # axes.set_xlim(-m, m); axes.set_ylim(-m, m)
        # axes.set_aspect('equal')
        # axes.set_xlabel('X'); axes.set_ylabel('Y')
        # axes.set_title('room outline (map0 + map1)')
        # axes.legend(); axes.grid(True)

        # plt.tight_layout()
        # plt.savefig('plane_dis_vis_overlay.png', dpi=150)
        # breakpoint()

        all_x = np.concatenate(all_x); all_y = np.concatenate(all_y)
        m = max(np.abs(all_x).max(), np.abs(all_y).max()) * 1.05
        for n, ax in enumerate(axes):
            ax.set_xlim(-m, m); ax.set_ylim(-m, m)
            ax.set_aspect('equal')
            ax.set_xlabel('X'); ax.set_ylabel('Y')
            ax.set_title(f'room outline (map{n})')
            ax.legend(); ax.grid(True)

        plt.tight_layout()
        plt.savefig('plane_dis_vis_side_by_side.png', dpi=150)
        # breakpoint()

    @torch.no_grad()
    def sample_pose(self, idx):
        pose = torch.eye(4)
        pose[:3, 3] = self.anchor_pts[idx]
        return pose



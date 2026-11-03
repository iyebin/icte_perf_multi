from .pose_sampler import PoseSampler

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d, minimum_filter1d

from utils.camera_utils import *


@torch.no_grad()
def _resample_uniformly(pts):
    """닫힌 궤적의 점들을 호 길이 기준으로 균일하게 다시 샘플링한다."""
    n = len(pts)
    pts_dense = F.interpolate(
        pts[None].permute(0, 2, 1),
        size=n * 128,
        mode="linear",
        align_corners=False,
    )[0].permute(1, 0)

    closed_pts = torch.cat([pts_dense, pts_dense[:1]], dim=0)
    segment_len = torch.linalg.norm(closed_pts[1:] - closed_pts[:-1], dim=-1)
    cumulative_len = torch.cumsum(segment_len, dim=0)
    cumulative_len = cumulative_len / cumulative_len[-1].clamp_min(1e-8)

    targets = torch.linspace(
        0.0,
        1.0 - 1.0 / n,
        n,
        device=pts.device,
        dtype=pts.dtype,
    )
    idx = torch.searchsorted(cumulative_len, targets).clamp_max(len(pts_dense) - 1)
    return pts_dense[idx]


@torch.no_grad()
def _get_trajectory_normals(pts):
    """XY 평면의 닫힌 궤적에 대한 부드러운 법선 벡터를 계산한다."""
    n_pts = len(pts)
    sigma = float(n_pts) / 32.0 * 2.0 + 1.0

    ext_pts = torch.cat([pts, pts[:1]], dim=0)
    right_vec = ext_pts[1:] - ext_pts[:-1]
    right_vec = right_vec / torch.linalg.norm(
        right_vec, dim=-1, keepdim=True
    ).clamp_min(1e-8)

    up_vec = torch.zeros_like(right_vec)
    up_vec[:, 2] = 1.0
    to_vec = torch.cross(up_vec, right_vec, dim=-1)
    to_vec = to_vec / torch.linalg.norm(
        to_vec, dim=-1, keepdim=True
    ).clamp_min(1e-8)

    to_vec_np = to_vec.cpu().numpy()
    for axis in range(3):
        to_vec_np[:, axis] = gaussian_filter1d(
            to_vec_np[:, axis], sigma=sigma, mode="wrap"
        )

    to_vec = torch.from_numpy(to_vec_np).to(device=pts.device, dtype=pts.dtype)
    to_vec = to_vec / torch.linalg.norm(
        to_vec, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    return -to_vec


class CirclePoseSampler(PoseSampler):
    """
    두 distance map의 방향별 최소 거리로 공통 안전 경계를 만든 뒤,
    그 경계 안쪽에 pose anchor를 생성한다.

    전제:
      1. 두 distance map의 해상도가 같다.
      2. 두 map의 수평 방향 인덱스가 같은 world azimuth를 뜻한다.
      3. 두 pose의 카메라 중심이 같다.

    주요 설정:
      - traverse_ratios: 0 < ratio <= 1. 값이 작을수록 더 안쪽이다.
      - safety_margin: 공통 최소 경계에서 먼저 빼는 절대 거리(optional).
      - trajectory_ratio: 법선 계산용 궤적의 내부 비율(optional, 기본 0.3).
      - visualize: 결과 그림 저장 여부(optional, 기본 True).
      - visualization_path: 저장 경로(optional).
    """

    def __init__(
        self,
        distance_map,
        poses,
        traverse_ratios,
        n_anchors_per_ratio,
        test_z_min_max=(0.0, 0.0),
        **kwargs,
    ):
        super().__init__()

        safety_margin = float(kwargs.pop("safety_margin", 0.0))
        trajectory_ratio = float(kwargs.pop("trajectory_ratio", 0.3))
        visualize = bool(kwargs.pop("visualize", True))
        visualization_path = kwargs.pop(
            "visualization_path", "plane_dis_vis_min_safe.png"
        )

        if len(distance_map) != 2 or len(poses) != 2:
            raise ValueError("distance_map과 poses는 각각 정확히 2개여야 합니다.")
        if len(traverse_ratios) != len(n_anchors_per_ratio):
            raise ValueError(
                "traverse_ratios와 n_anchors_per_ratio의 길이가 같아야 합니다."
            )
        if not traverse_ratios:
            raise ValueError("traverse_ratios가 비어 있으면 안 됩니다.")
        if any(r <= 0.0 or r > 1.0 for r in traverse_ratios):
            raise ValueError(
                "안쪽 pose 생성을 위해 모든 traverse_ratio는 0 < ratio <= 1이어야 합니다."
            )
        if any(k <= 0 for k in n_anchors_per_ratio):
            raise ValueError("모든 anchor 개수는 1 이상이어야 합니다.")
        if safety_margin < 0.0:
            raise ValueError("safety_margin은 0 이상이어야 합니다.")
        if trajectory_ratio <= 0.0 or trajectory_ratio > 1.0:
            raise ValueError("trajectory_ratio는 0 < ratio <= 1이어야 합니다.")

        device = poses[0].device
        dtype = poses[0].dtype
        centers = [pose[:3, 3].to(device=device, dtype=dtype) for pose in poses]

        # 단순한 방향별 min은 두 map의 원점과 azimuth 정렬이 같을 때만 유효하다.
        if not torch.allclose(centers[0], centers[1], atol=1e-4, rtol=0.0):
            raise ValueError(
                "두 카메라 중심이 다릅니다. 현재 방식으로는 방향별 min을 직접 계산할 "
                "수 없습니다. 두 map을 같은 world 중심으로 재투영해야 합니다."
            )

        center = centers[0]
        filtered_dis_list = []
        raw_dis_list = []
        azimuth_list = []
        map_center_np = []
        widths = []

        # map별 디버깅/시각화 데이터를 덮어쓰지 않도록 리스트로 보관한다.
        self.plane_pts_raw = []
        self.plane_pts_filter = []
        self.plane_pts_smooth = []

        circle_pts = None

        for n in range(2):
            distance = distance_map[n]
            if torch.is_tensor(distance):
                distance = distance.detach().cpu().numpy()
            distance = np.asarray(distance).squeeze()

            if distance.ndim != 2:
                raise ValueError(
                    f"distance_map[{n}]은 squeeze 후 2차원이어야 합니다: "
                    f"현재 shape={distance.shape}"
                )

            height, width = distance.shape
            widths.append(width)
            if n == 1 and widths[0] != width:
                raise ValueError("두 distance map의 width가 같아야 합니다.")

            pano_coords = img_to_pano_coord(img_coord_from_hw(height, width))
            pano_coords_np = pano_coords.detach().cpu().numpy()

            # 적도 부근의 depth를 수평면 거리로 투영하고, 각 azimuth의 보수적인
            # 최소값을 선택한다.
            plane_dis_rows = distance * np.cos(pano_coords_np[:, :, 0])
            half_band = min(10, height // 2)
            h_center = height // 2
            row_start = max(0, h_center - half_band)
            row_end = min(height, h_center + half_band)
            plane_dis_rows = plane_dis_rows[row_start:row_end]
            plane_dis_rows[plane_dis_rows < 1e-5] = 1e9
            plane_dis = np.min(plane_dis_rows, axis=0)

            # 좌우 방향으로 invalid 값을 채운다.
            for i in range(1, width):
                if plane_dis[i] > 1e8:
                    plane_dis[i] = plane_dis[i - 1]
            for i in range(1, width):
                idx = width - i - 1
                if plane_dis[idx] > 1e8:
                    plane_dis[idx] = plane_dis[idx + 1]

            if np.all(plane_dis > 1e8):
                raise ValueError(f"distance_map[{n}]에서 유효한 거리를 찾지 못했습니다.")

            pool_size = max(1, (width // 16) // 2 * 2 + 1)
            conservative_dis = minimum_filter1d(
                plane_dis, size=pool_size, mode="wrap"
            )

            smooth_sigma = max(1, (width // 8) // 2 * 2 + 1)
            smoothed_dis = gaussian_filter1d(
                conservative_dis, sigma=smooth_sigma, mode="wrap"
            )

            blur_sigma = max(1, (width // 64) // 2 * 2 + 1)
            filtered_dis = gaussian_filter1d(
                conservative_dis, sigma=blur_sigma, mode="wrap"
            )

            # Gaussian blur가 minimum-filter 경계보다 바깥으로 나가지 않게 제한한다.
            filtered_dis = np.minimum(filtered_dis, conservative_dis)
            smoothed_dis = np.minimum(smoothed_dis, filtered_dis)

            plane_coords = torch.stack(
                [
                    torch.full((width,), 0.5),
                    torch.linspace(0.5 / width, 1.0 - 0.5 / width, width),
                ],
                dim=-1,
            )
            cur_circle_pts = img_coord_to_pano_direction(plane_coords).to(
                device=device, dtype=dtype
            )

            if circle_pts is None:
                circle_pts = cur_circle_pts

            filtered_dis_t = torch.as_tensor(
                filtered_dis, device=device, dtype=dtype
            )
            t = centers[n]

            self.plane_pts_raw.append(
                cur_circle_pts
                * torch.as_tensor(plane_dis, device=device, dtype=dtype)[:, None]
                + t
            )
            self.plane_pts_filter.append(
                cur_circle_pts * filtered_dis_t[:, None] + t
            )
            self.plane_pts_smooth.append(
                cur_circle_pts
                * torch.as_tensor(smoothed_dis, device=device, dtype=dtype)[:, None]
                + t
            )

            raw_dis_list.append(plane_dis.copy())
            filtered_dis_list.append(filtered_dis_t)
            azimuth_list.append(pano_coords_np[0, :, 1].copy())
            map_center_np.append(t.detach().cpu().numpy())

        # 핵심: 각 azimuth에서 두 map 중 더 가까운 경계를 공통 안전 경계로 사용한다.
        common_safe_dis = torch.stack(filtered_dis_list, dim=0).amin(dim=0)
        common_safe_dis = (common_safe_dis - safety_margin).clamp_min(1e-4)
        self.common_safe_dis = common_safe_dis

        test_z_min, test_z_max = test_z_min_max
        anchor_pts = []
        anchor_groups = []

        for i, (traverse_ratio, k) in enumerate(
            zip(traverse_ratios, n_anchors_per_ratio)
        ):
            # ratio < 1이면 두 map의 공통 최소 경계보다 안쪽에 생성된다.
            traverse_pts = (
                circle_pts * common_safe_dis[:, None] * traverse_ratio + center
            )
            traverse_pts = _resample_uniformly(traverse_pts)

            bias = 0.0 if i % 2 == 0 else 0.5 / k
            anchor_fraction = torch.linspace(
                0.5 / k,
                1.0 - 0.5 / k,
                k,
                device=device,
                dtype=dtype,
            ) + bias
            anchor_idx = (anchor_fraction * len(traverse_pts)).long()
            anchor_idx = anchor_idx.clamp(0, len(traverse_pts) - 1)

            cur_pts = traverse_pts[anchor_idx].clone()
            for j in range(len(cur_pts)):
                cur_pts[j, 2] = (
                    test_z_min if (i + j) % 2 == 0 else test_z_max
                )

            anchor_pts.append(cur_pts)
            anchor_groups.append(cur_pts.detach().cpu().numpy())

        self.anchor_pts = torch.cat(anchor_pts, dim=0)

        # 법선 계산용 궤적 역시 common_safe_dis 안쪽에서 만든다.
        trajectory_dis = common_safe_dis * trajectory_ratio
        self.traverse_pts = _resample_uniformly(
            circle_pts * trajectory_dis[:, None] + center
        )
        self.traverse_pts[:, 2] = (test_z_min + test_z_max) * 0.5
        self.traverse_normals = _get_trajectory_normals(self.traverse_pts)
        self.n_anchors = len(self.anchor_pts)
        self.n_poses = self.n_anchors

        if visualize:
            self._save_visualization(
                raw_dis_list=raw_dis_list,
                filtered_dis_list=filtered_dis_list,
                azimuth_list=azimuth_list,
                map_center_np=map_center_np,
                anchor_groups=anchor_groups,
                traverse_ratios=traverse_ratios,
                output_path=visualization_path,
            )

    @torch.no_grad()
    def _save_visualization(
        self,
        raw_dis_list,
        filtered_dis_list,
        azimuth_list,
        map_center_np,
        anchor_groups,
        traverse_ratios,
        output_path,
    ):
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        colors = ["#1f77b4", "#ff7f0e"]
        all_x, all_y = [], []

        for n in range(2):
            t = map_center_np[n]
            azimuth = azimuth_list[n]
            filtered_np = filtered_dis_list[n].detach().cpu().numpy()

            for dis, color, alpha, size, label in [
                (raw_dis_list[n], "#9e9e9e", 0.45, 3, f"map{n} raw"),
                (filtered_np, colors[n], 0.75, 8, f"map{n} filtered"),
            ]:
                valid = dis < 1e8
                x = dis[valid] * np.cos(azimuth[valid]) + t[0]
                y = dis[valid] * np.sin(azimuth[valid]) + t[1]
                ax.scatter(x, y, s=size, c=color, alpha=alpha, label=label)
                all_x.append(x)
                all_y.append(y)

        # 공통 min 안전 경계
        safe_np = self.common_safe_dis.detach().cpu().numpy()
        t = map_center_np[0]
        azimuth = azimuth_list[0]
        safe_x = safe_np * np.cos(azimuth) + t[0]
        safe_y = safe_np * np.sin(azimuth) + t[1]
        ax.plot(safe_x, safe_y, c="black", lw=2.0, label="common min boundary")
        all_x.append(safe_x)
        all_y.append(safe_y)

        linestyles = ["-", "--", ":", "-."]
        for i, group in enumerate(anchor_groups):
            gx = np.append(group[:, 0], group[0, 0])
            gy = np.append(group[:, 1], group[0, 1])
            ax.plot(
                gx,
                gy,
                ls=linestyles[i % len(linestyles)],
                c="#2ca02c",
                lw=1.5,
                label=f"anchors ratio={traverse_ratios[i]}",
            )
            ax.scatter(
                group[:, 0],
                group[:, 1],
                s=55,
                c="#2ca02c",
                edgecolors="black",
                linewidths=0.7,
                zorder=6,
            )
            all_x.append(group[:, 0])
            all_y.append(group[:, 1])

        ax.scatter(t[0], t[1], c="red", marker="*", s=170, label="camera")
        all_x.append(np.asarray([t[0]]))
        all_y.append(np.asarray([t[1]]))

        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)
        x_mid = 0.5 * (all_x.min() + all_x.max())
        y_mid = 0.5 * (all_y.min() + all_y.max())
        half_range = 0.525 * max(all_x.max() - all_x.min(), all_y.max() - all_y.min())
        half_range = max(half_range, 1e-3)

        ax.set_xlim(x_mid - half_range, x_mid + half_range)
        ax.set_ylim(y_mid - half_range, y_mid + half_range)
        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("room outline: common min safe boundary and inner poses")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    @torch.no_grad()
    def sample_pose(self, idx):
        pose = torch.eye(
            4,
            device=self.anchor_pts.device,
            dtype=self.anchor_pts.dtype,
        )
        pose[:3, 3] = self.anchor_pts[idx]
        return pose

"""
5개(N개) PanoVGGT pose 각각을 기준으로 merged point cloud 를 ERP distance map 으로 렌더링해 시각화.

사용:
    python vis_pose_distance.py <image_dir> [--H 518 --W 1036 --tol 0.03 --fill 3]

입력 (PanoVggt.main 이 이미 저장한 파일):
    <image_dir>/ref_geometry/ref_geometry.ply   merged world points (+rgb)
    <image_dir>/all_poses.npy                   (S, 4, 4) PanoVGGT poses
    <image_dir>/ref_geometry/ref_transform.npz  (선택) 검증된 pose 해석 / R_cam2perf

출력: <image_dir>/pose_distance_vis/
    dist_pose{i}.png     pose i 기준 distance map (전 pose 공통 color scale, log)
    dist_pose{i}.npy     raw distance (invalid = 0)
    montage.png          전체 세로 이어붙이기
    각 이미지 위에 다른 카메라 위치를 흰 원 + 번호로 표시
"""
import os
import sys
import argparse

import cv2
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from utils.camera_utils import direction_to_img_coord  # noqa: E402

R_FIXED = np.array([[0, 0, 1],
                    [-1, 0, 0],
                    [0, -1, 0]], dtype=np.float64)  # OpenCV cam -> PeRF pano


# ----------------------------------------------------------------------
def load_ply_xyz(path):
    """PanoVggt.save_ply 형식(binary little endian, xyz float + rgb uchar) 로드."""
    with open(path, "rb") as f:
        n = None
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            if line == "end_header":
                break
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                       ("r", "u1"), ("g", "u1"), ("b", "u1")])
        data = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
    return np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)


def resolve_convention(image_dir, poses):
    """ref_transform.npz 로 pose 해석(c2w/w2c)과 R_cam2perf 결정."""
    path = os.path.join(image_dir, "ref_geometry", "ref_transform.npz")
    if not os.path.exists(path):
        print(f"[vis][warn] {path} 없음 → c2w + OpenCV->PeRF 가정")
        return "c2w", R_FIXED
    T = np.load(path)
    ref_c2w, R = T["ref_c2w"].astype(np.float64), T["R_cam2perf"].astype(np.float64)
    if np.allclose(ref_c2w, poses[0], atol=1e-4):
        conv = "c2w"
    elif np.allclose(ref_c2w, np.linalg.inv(poses[0]), atol=1e-4):
        conv = "w2c"
    else:
        print("[vis][warn] ref_c2w 가 pose[0] 과 일치하지 않음 → c2w 가정")
        conv = "c2w"
    print(f"[vis] pose convention = {conv}")
    print(f"[vis] R_cam2perf =\n{np.round(R, 3)}")
    return conv, R


def world_to_pixel(pts_world, w2c, R, H, W):
    """world points -> (v, u, distance, valid)."""
    cam = pts_world @ w2c[:3, :3].T + w2c[:3, 3]
    d = np.linalg.norm(cam, axis=1)
    ok = np.isfinite(cam).all(axis=1) & (d > 1e-6)
    dirs = (cam[ok] / d[ok, None]) @ R.T
    coord = direction_to_img_coord(torch.from_numpy(dirs.astype(np.float32))).cpu().numpy()
    v = np.clip(np.floor(coord[:, 0] * H).astype(np.int64), 0, H - 1)
    u = np.floor(coord[:, 1] * W).astype(np.int64) % W
    return v, u, d[ok]


def render_distance(pts_world, w2c, R, H, W, tol, fill_iters):
    """merged_distance_map 과 같은 규칙: 앞 surface 평균 + floater 제거 + 구멍 채우기."""
    v, u, d = world_to_pixel(pts_world, w2c, R, H, W)
    pix = v * W + u

    d_min = np.full(H * W, np.inf)
    np.minimum.at(d_min, pix, d)
    front = d <= d_min[pix] * (1.0 + tol)
    s = np.bincount(pix[front], weights=d[front], minlength=H * W)
    c = np.bincount(pix[front], minlength=H * W)
    out = np.zeros(H * W)
    has = c > 0
    out[has] = s[has] / c[has]
    out, valid = out.reshape(H, W), has.reshape(H, W)

    pad = np.pad(np.where(valid, out, np.nan), 1, mode="wrap")
    win = np.lib.stride_tricks.sliding_window_view(pad, (3, 3)).reshape(H, W, 9)
    with np.errstate(all="ignore"):
        med = np.nanmedian(win, axis=-1)
    floater = valid & np.isfinite(med) & (out < med * (1.0 - 5 * tol))
    out[floater] = med[floater]

    for _ in range(fill_iters):
        holes = ~valid
        if not holes.any():
            break
        pd = np.pad(np.where(valid, out, 0.0), 1, mode="wrap")
        pm = np.pad(valid.astype(np.float64), 1, mode="wrap")
        ss = np.lib.stride_tricks.sliding_window_view(pd, (3, 3)).sum(axis=(-1, -2))
        cc = np.lib.stride_tricks.sliding_window_view(pm, (3, 3)).sum(axis=(-1, -2))
        f = holes & (cc > 0)
        out[f] = ss[f] / cc[f]
        valid |= f

    out[~valid] = 0.0
    return out.astype(np.float32)


def colorize(d, lo, hi):
    valid = d > 0
    x = np.clip(d, lo, hi)
    vis = (np.log(x) - np.log(lo)) / (np.log(hi) - np.log(lo))
    img = cv2.applyColorMap((np.clip(vis, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~valid] = 0
    return img


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir")
    ap.add_argument("--H", type=int, default=518)
    ap.add_argument("--W", type=int, default=1036)
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--fill", type=int, default=3)
    args = ap.parse_args()

    image_dir, H, W = args.image_dir, args.H, args.W
    out_dir = os.path.join(image_dir, "pose_distance_vis")
    os.makedirs(out_dir, exist_ok=True)

    pts = load_ply_xyz(os.path.join(image_dir, "ref_geometry", "ref_geometry.ply"))
    poses = np.load(os.path.join(image_dir, "all_poses.npy")).astype(np.float64)
    S = poses.shape[0]
    print(f"[vis] {pts.shape[0]:,} points, {S} poses")

    conv, R = resolve_convention(image_dir, poses)
    c2ws = poses if conv == "c2w" else np.linalg.inv(poses)
    centers = c2ws[:, :3, 3]

    # ---- 각 pose 기준 distance map ----
    dists = []
    for i in range(S):
        w2c = np.linalg.inv(c2ws[i])
        d = render_distance(pts, w2c, R, H, W, args.tol, args.fill)
        np.save(os.path.join(out_dir, f"dist_pose{i}.npy"), d)
        v = d[d > 0]
        print(f"[vis] pose {i}: center={np.round(centers[i], 3)}  valid={np.mean(d > 0):.4f}  "
              f"min={v.min():.3f}  median={np.median(v):.3f}  max={v.max():.3f}")
        dists.append(d)

    # ---- 전 pose 공통 color scale ----
    allv = np.concatenate([d[d > 0] for d in dists])
    lo, hi = np.percentile(allv, [2, 98])
    print(f"[vis] common color range (log): [{lo:.3f}, {hi:.3f}]")

    tiles = []
    for i, d in enumerate(dists):
        img = colorize(d, lo, hi)

        # 다른 카메라 위치 표시
        w2c = np.linalg.inv(c2ws[i])
        for j in range(S):
            if j == i:
                continue
            vj, uj, dj = world_to_pixel(centers[j][None], w2c, R, H, W)
            if len(vj) == 0:
                continue
            p = (int(uj[0]), int(vj[0]))
            cv2.circle(img, p, 7, (255, 255, 255), 2)
            cv2.putText(img, f"{j} ({dj[0]:.1f})", (p[0] + 9, p[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(img, f"pose {i}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, f"dist_pose{i}.png"), img)
        tiles.append(img)

    cv2.imwrite(os.path.join(out_dir, "montage.png"), np.concatenate(tiles, axis=0))
    print(f"[vis] saved → {out_dir}")


if __name__ == "__main__":
    main()

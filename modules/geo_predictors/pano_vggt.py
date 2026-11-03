import os
# import argparse
import contextlib
import glob
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from .panovggt_model import PanoVGGTModel
from utils.camera_utils import direction_to_img_coord

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

_INPUT_H = 518
_INPUT_W = 1036

class PanoVggt:

    def __init__(
        self,
        device: str = "cuda",
    ):
        
        self.device = device

        self.config_path = "configs/default.yaml"
        self.checkpoint_path = "pre_checkpoints/panovggt_model.pt"

        self.model = self.load_model(
            self.config_path,
            self.checkpoint_path,
            self.device
        )
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        self.model.eval().to(device)
        
    def load_model(self, config_path: str, checkpoint_path: str, device: str) -> PanoVGGTModel:
        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        mc = cfg.model
        model = PanoVGGTModel(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
            enable_camera=mc.enable_camera,
            enable_depth=mc.enable_depth,
            enable_point=mc.enable_point,
            aggregator=OmegaConf.to_container(mc.aggregator, resolve=True),
        )
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        for key in ("model_state_dict", "model", "state_dict"):
            if key in ckpt:
                ckpt = ckpt[key]
                break
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[load] missing keys  : {missing[:5]}{'…' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"[load] unexpected keys: {unexpected[:5]}{'…' if len(unexpected) > 5 else ''}")
        print("✅ PanoVGGT model loaded successfully.")
        return model

    def collect_images(self, image_dir: str) -> List[str]:
        """Return sorted image paths from a directory."""
        paths = sorted(
            p for p in glob.glob(os.path.join(image_dir, "*"))
            if os.path.splitext(p)[1].lower() in _IMG_EXTS
        )
        if not paths:
            raise ValueError(f"No images found in: {image_dir}")
        return paths

    def load_images_fixed(self, image_paths: List[str]) -> torch.Tensor:
        """
        Load images, resize each to (_INPUT_H, _INPUT_W), normalise to [0,1].
        Returns float32 tensor of shape (S, 3, _INPUT_H, _INPUT_W).
        """
        frames = []
        for p in image_paths:
            bgr = cv2.imread(p)
            if bgr is None:
                raise IOError(f"Cannot read image: {p}")
            bgr = cv2.resize(bgr, (_INPUT_W, _INPUT_H), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            # (H, W, 3) → (3, H, W)
            frames.append(torch.from_numpy(rgb).permute(2, 0, 1))
        return torch.stack(frames, dim=0)  # (S, 3, H, W)

    def depth_to_colormap(self, 
        depth: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        colormap: int = cv2.COLORMAP_TURBO,
        use_log: bool = True,
    ) -> np.ndarray:
        """
        Convert a (H, W) float depth array to a uint8 BGR colour image.
        Invalid pixels (mask == False) are rendered black.
        """
        d = depth.copy().astype(np.float32)

        if valid_mask is not None:
            d[~valid_mask] = np.nan

        if use_log:
            d = np.log1p(d)

        d_min = np.nanmin(d)
        d_max = np.nanmax(d)
        if d_max - d_min < 1e-6:
            d_norm = np.zeros_like(d, dtype=np.uint8)
        else:
            d_norm = ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)

        d_norm = np.nan_to_num(d_norm, nan=0).astype(np.uint8)
        colored = cv2.applyColorMap(d_norm, colormap)

        if valid_mask is not None:
            colored[~valid_mask] = 0

        return colored

    def points_and_colors_from_frame(self,
        points_hw3: np.ndarray,
        image_hw3: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Flatten (H, W, 3) points and image arrays, applying optional mask.
        Returns (N, 3) xyz and (N, 3) rgb uint8.
        """
        H, W, _ = points_hw3.shape
        mask = valid_mask if valid_mask is not None else np.ones((H, W), dtype=bool)

        # discard zero-depth / NaN points
        depth = np.linalg.norm(points_hw3, axis=-1)
        mask = mask & (depth > 0) & np.isfinite(depth)

        xyz = points_hw3[mask]  # (N, 3)
        if image_hw3.max() <= 1.0:
            rgb = (image_hw3[mask] * 255).astype(np.uint8)
        else:
            rgb = image_hw3[mask].astype(np.uint8)

        return xyz, rgb

    def save_ply(self, path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
        """Write a coloured point cloud to a PLY file (binary little-endian for speed)."""
        assert xyz.shape[0] == rgb.shape[0], "xyz / rgb length mismatch"
        N = xyz.shape[0]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        xyz = xyz.astype(np.float32)
        rgb = rgb.astype(np.uint8)

        with open(path, "wb") as f:
            header = (
                "ply\n"
                "format binary_little_endian 1.0\n"
                f"element vertex {N}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property uchar red\n"
                "property uchar green\n"
                "property uchar blue\n"
                "end_header\n"
            )
            f.write(header.encode("ascii"))
            # interleave xyz (float32 × 3) and rgb (uint8 × 3) per vertex
            data = np.empty(N, dtype=[
                ("x", np.float32), ("y", np.float32), ("z", np.float32),
                ("r", np.uint8),   ("g", np.uint8),   ("b", np.uint8),
            ])
            data["x"] = xyz[:, 0]
            data["y"] = xyz[:, 1]
            data["z"] = xyz[:, 2]
            data["r"] = rgb[:, 0]
            data["g"] = rgb[:, 1]
            data["b"] = rgb[:, 2]
            f.write(data.tobytes())

        print(f"  [ply] saved {N:,} points → {path}")


    def run_inference(self,
        model: PanoVGGTModel,
        image_paths,
        device: str,
    ) -> dict:
        """
        Run PanoVGGT on the given image paths.
        image_paths: [ "data/image/000.jpg", "data/image/001.jpg"] ✅os.path.join(img_dir, img_names[i])
        Images are resized to (_INPUT_H, _INPUT_W) = (518, 1036) before inference.
        Returns a dict of numpy arrays (batch dim already squeezed).
        """
        # image_paths = [os.path.join(img_dir, img_name[i]) for i in range(len(img_name))]
        imgs = self.load_images_fixed(image_paths).to(device)
        print(f"[infer] Input tensor shape: {imgs.shape}  "
            f"(S={imgs.shape[0]}, C=3, H={_INPUT_H}, W={_INPUT_W})")

        # autocast only on CUDA
        amp_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if device == "cuda"
            else contextlib.nullcontext()
        )

        with torch.no_grad(), amp_ctx:
            # imgs: (S, 3, H, W) → add batch dim → (1, S, 3, H, W)
            preds = model(imgs.unsqueeze(0))

        # Convert to float32 CPU numpy, squeeze batch dim
        out: dict = {}
        for k, v in preds.items():
            if isinstance(v, torch.Tensor):
                v_f = v.float() if v.dtype == torch.bfloat16 else v
                v_cpu = v_f.cpu()
                # squeeze leading batch dim if present
                if v_cpu.dim() >= 1 and v_cpu.shape[0] == 1:
                    v_cpu = v_cpu.squeeze(0)
                out[k] = v_cpu.numpy()
            else:
                out[k] = v

        return out

    # def merged_distance_map(self, merged_xyz, ref_pose, H=518, W=1036,
    #                     ref_points_hw3=None, transform_path=None,
    #                     surface_tol=0.03, fill_iters=3):

    #     """
    #     merged_xyz : (N, 3)
    #         PanoVGGT world-frame points

    #     ref_pose : (4, 4)
    #         PanoVGGT camera-to-world pose (OpenCV c2w)

    #     return:
    #         merged_distance : (H, W)
    #         reference panorama 기준 Euclidean distance map
    #     """

    #     assert merged_xyz.ndim == 2
    #     assert merged_xyz.shape[1] == 3
    #     assert ref_pose.shape == (4, 4)

    #     # --------------------------------------------------
    #     # 1. world -> reference camera
    #     #
    #     # PanoVGGT ref_pose = c2w
    #     # therefore inverse(ref_pose) = w2c
    #     # --------------------------------------------------
    #     ref_w2c = np.linalg.inv(ref_pose)

    #     ones = np.ones(
    #         (merged_xyz.shape[0], 1),
    #         dtype=merged_xyz.dtype
    #     )

    #     points_h = np.concatenate(
    #         [merged_xyz, ones],
    #         axis=1
    #     )

    #     points_cam_h = (
    #         ref_w2c @ points_h.T
    #     ).T

    #     # OpenCV camera coordinates
    #     points_cv = points_cam_h[:, :3]

    #     # --------------------------------------------------
    #     # 2. distance from reference camera
    #     #
    #     # Euclidean/ray distance
    #     # --------------------------------------------------
    #     distance = np.linalg.norm(
    #         points_cv,
    #         axis=1
    #     )

    #     valid = (
    #         np.isfinite(points_cv).all(axis=1)
    #         & np.isfinite(distance)
    #         & (distance > 1e-6)
    #     )

    #     points_cv = points_cv[valid]
    #     distance = distance[valid]

    #     # --------------------------------------------------
    #     # 3. OpenCV camera coordinates
    #     #          ->
    #     #    PeRF panorama local coordinates
    #     #
    #     # OpenCV:
    #     #   +X = right
    #     #   +Y = down
    #     #   +Z = forward
    #     #
    #     # PeRF pano:
    #     #   +X = forward
    #     #   -Y = right
    #     #   -Z = down
    #     # --------------------------------------------------
    #     points_perf = np.stack(
    #     [
    #         points_cv[:, 2],    # PeRF X = OpenCV Z
    #         -points_cv[:, 0],   # PeRF Y = -OpenCV X
    #         -points_cv[:, 1],   # PeRF Z = -OpenCV Y
    #     ], axis=1)

    #     # --------------------------------------------------
    #     # 4. points -> unit directions
    #     # --------------------------------------------------
    #     dirs_perf = (
    #         points_perf
    #         / distance[:, None]
    #     )

    #     # --------------------------------------------------
    #     # 5. direction -> ERP image coordinates
    #     #
    #     # PeRF 자체 convention 사용
    #     # output:
    #     #   img_coord[:, 0] = normalized row [0,1]
    #     #   img_coord[:, 1] = normalized col [0,1]
    #     # --------------------------------------------------

    #     dirs_t = torch.from_numpy(
    #     dirs_perf.astype(np.float32)
    #     )

    #     img_coord = (
    #         direction_to_img_coord(dirs_t)
    #         .cpu()
    #         .numpy()
    #     )

    #     v = np.floor(
    #         img_coord[:, 0] * H
    #     ).astype(np.int32)

    #     u = np.floor(
    #         img_coord[:, 1] * W
    #     ).astype(np.int32)


    #     # ERP horizontal seam: wrap around
    #     u = u % W

    #     # top / bottom
    #     v = np.clip(
    #         v,
    #         0,
    #         H - 1
    #     )

    #     # --------------------------------------------------
    #     # 6. spherical z-buffer
    #     #
    #     # 같은 ERP pixel에 여러 point가 투영되면
    #     # reference camera에서 가장 가까운 point 선택
    #     # --------------------------------------------------

    #     merged_distance = np.full(
    #     (H, W),
    #     np.inf,
    #     dtype=np.float32
    #     )

    #     np.minimum.at(
    #         merged_distance,
    #         (v, u),
    #         distance.astype(np.float32)
    #     )

    #     # 7. validity statistics
    #     valid_pixels = np.isfinite(
    #         merged_distance
    #     )

    #     valid_ratio = valid_pixels.mean()

    #     print(
    #         f"valid pixel ratio: "
    #         f"{valid_ratio:.4f} "
    #         f"({valid_pixels.sum()}/{H * W})"
    #     )

    #     # Empty pixels -> 0
    #     merged_distance[
    #         ~valid_pixels
    #     ] = 0.0

    #     return merged_distance

    def main(self, image_dir, image_names):
        ''' 
        return: distances, normals
        '''
        # ── device ────────────────────────────────────────────────────────────
        # device = args.device
        # device = self.device
        if self.device == "cuda" and not torch.cuda.is_available():
            print("[warn] CUDA not available, falling back to CPU.")
            self.device = "cpu"

        image_paths = [os.path.join(image_dir, image_names[i]) for i in range(len(image_names))]
        
        # ── output dirs ───────────────────────────────────────────────────────
        # out_root      = args.output_dir
        # depth_dir     = os.path.join(out_root, "depth")
        # pose_dir      = os.path.join(out_root, "poses")
        # per_frame_dir = os.path.join(out_root, "pointclouds", "per_frame")
        # merged_dir    = os.path.join(out_root, "pointclouds")
        # for d in (depth_dir, pose_dir, per_frame_dir, merged_dir):
        #     os.makedirs(d, exist_ok=True)

        # ── collect inputs ────────────────────────────────────────────────────
        # image_paths = self.collect_images(args.image_dir)
        # mask_paths  = collect_masks(args.mask_dir, image_paths)
        # S = len(image_paths)
        # print(f"[pipeline] {S} image(s) found.")
        # if mask_paths is not None:
        #     n_masks = sum(1 for m in mask_paths if m is not None)
        #     print(f"[pipeline] {n_masks}/{S} mask(s) matched.")

        # ── model ─────────────────────────────────────────────────────────────
        # model = self.load_model(args.config, args.checkpoint, device)
        # self.config_path = os.path.join("configs", "default.yaml")
        # self.model_path = os.path.join("pre_checkpoints", "model.pt")
        # model = self.load_model(self.config_path, self.model_path, device)
        # self.model.eval().to(device)
        print(f"[pipeline] Model ready on {self.device}.")
        print(f"[pipeline] Fixed input resolution: H={_INPUT_H}, W={_INPUT_W}")

        # ── inference ─────────────────────────────────────────────────────────
        preds = self.run_inference(self.model, image_paths, self.device)

        # ── unpack predictions ────────────────────────────────────────────────
        # After squeeze, expected shapes:
        #   depth        : (S, H, W) or (S, H, W, 1)
        #   local_points : (S, H, W, 3)   — camera / local frame
        #   world_points : (S, H, W, 3)   — world frame  (preferred for merged cloud)
        #   camera_poses : (S, 4, 4)
        #   images       : (S, C, H, W) or (S, H, W, C)  — model's view of the input

        # depth  (S, H, W)
        depth_np: Optional[np.ndarray] = None
        if "depth" in preds and preds["depth"] is not None:
            depth_np = preds["depth"]
            if depth_np.ndim == 4:          # (S, H, W, 1) → (S, H, W)
                depth_np = depth_np[..., 0]

        # world-frame points  (S, H, W, 3)
        world_pts_np: Optional[np.ndarray] = None
        if "world_points" in preds and preds["world_points"] is not None:
            world_pts_np = preds["world_points"]
        elif "points" in preds and preds["points"] is not None:
            world_pts_np = preds["points"]

        # local-frame points  (S, H, W, 3)
        local_pts_np: Optional[np.ndarray] = None
        if "local_points" in preds and preds["local_points"] is not None:
            local_pts_np = preds["local_points"]
        elif world_pts_np is not None:
            local_pts_np = world_pts_np     # fall back

        # camera poses  (S, 4, 4)
        poses_np: Optional[np.ndarray] = None
        if "camera_poses" in preds and preds["camera_poses"] is not None:
            poses_np = preds["camera_poses"]

        # H, W at inference resolution
        H, W = _INPUT_H, _INPUT_W

        print(f"[pipeline] Inference frame size: {H} × {W}")

        # ── per-frame processing ──────────────────────────────────────────────
        all_xyz: List[np.ndarray] = []
        all_rgb: List[np.ndarray] = []

        
        for i, img_path in enumerate(image_paths):
            stem = Path(img_path).stem
            print(f"\n[frame {i:04d}] {stem}")

            # Load original image, resize to inference resolution for colour lookup
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"  [warn] Cannot read image: {img_path}. Skipping.")
                continue
            img_bgr_resized = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_AREA)
            img_rgb = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # Load and resize mask to inference resolution
            mask_valid: Optional[np.ndarray] = None
            # if mask_paths is not None and mask_paths[i] is not None:
            #     mask_valid = load_mask(mask_paths[i], H, W)
            #     if mask_valid is not None:
            #         print(f"  mask: {mask_paths[i]}  "
            #             f"valid={mask_valid.sum():,}/{H * W:,} px "
            #             f"({100.0 * mask_valid.mean():.1f} %)")
            # else:
            #     print("  mask: (none — all pixels valid)")

            # ── depth map ─────────────────────────────────────────────────────
            if depth_np is not None:
                d = depth_np[i]                                 # (H, W)

                # coloured visualisation
                colored = self.depth_to_colormap(
                    d,
                    valid_mask=mask_valid,
                    use_log=True,
                    colormap=cv2.COLORMAP_TURBO,
                )
                out_depth_path = os.path.join(image_dir, f"{stem}_depth.png")
                cv2.imwrite(out_depth_path, colored)

                # raw 16-bit depth (millimetres, masked invalid → 0)
                d_mm = (d * 1000.0).astype(np.float32)
                if mask_valid is not None:
                    d_mm[~mask_valid] = 0.0
                d_u16 = np.clip(d_mm, 0, 65535).astype(np.uint16)
                cv2.imwrite(
                    os.path.join(image_dir, f"{stem}_depth_raw.png"), d_u16
                )
                print(f"  depth: saved → {out_depth_path}")

            # ── camera pose ───────────────────────────────────────────────────
            if poses_np is not None:
                pose44 = poses_np[i]                            # (4, 4)
                R = pose44[:3, :3]                              # (3, 3)
                t = pose44[:3, 3]                               # (3,)

                np.savetxt(
                    os.path.join(image_dir, f"{stem}_R.txt"), R,
                    fmt="%.8f",
                    header=f"Rotation matrix for frame {i}: {stem}",
                )
                np.savetxt(
                    os.path.join(image_dir, f"{stem}_t.txt"), t[np.newaxis],
                    fmt="%.8f",
                    header=f"Translation for frame {i}: {stem}",
                )
                np.save(os.path.join(image_dir, f"{stem}_pose.npy"), pose44)
                print(f"  pose: R saved  t={t}")

            # ── per-frame point cloud (local / camera frame) ──────────────────
            if local_pts_np is not None:
                pts_hw3 = local_pts_np[i]                       # (H, W, 3)
                xyz, rgb = self.points_and_colors_from_frame(
                    pts_hw3, img_rgb, valid_mask=mask_valid
                )
                ply_path = os.path.join(image_dir, f"{stem}.ply")
                self.save_ply(ply_path, xyz, rgb)

            # ── accumulate world-frame points for merged cloud ─────────────────
            if world_pts_np is not None:
                pts_hw3_w = world_pts_np[i]                     # (H, W, 3)
                xyz_w, rgb_w = self.points_and_colors_from_frame(
                    pts_hw3_w, img_rgb, valid_mask=mask_valid
                )
                all_xyz.append(xyz_w)
                all_rgb.append(rgb_w)

        # ── merged point cloud ────────────────────────────────────────────────
        global_max = max(d.max().item() for d in depth_np)
        scale = global_max * 1.05
        
        if all_xyz:
            merged_xyz = np.concatenate(all_xyz, axis=0)
            merged_rgb = np.concatenate(all_rgb, axis=0)

            world_scale = float(np.abs(merged_xyz).max()) * 1.05

            # Depth 정규화
            # breakpoint() #depth/pose/merged_xyz가 제대로 normalization 됐는지 확인하기
            depth_np = depth_np / world_scale

            # PanoVGGT world point cloud
            norm_xyz = merged_xyz / world_scale

            self.save_ply(os.path.join(image_dir, "ref_geometry", "ref_geometry.ply"), norm_xyz, merged_rgb)
            print(f"\n[pipeline] Merged cloud: {len(norm_xyz):,} points total.")

        # ── save all poses together ───────────────────────────────────────────
        if poses_np is not None:
            np.save(os.path.join(image_dir, "all_poses.npy"), poses_np)
            np.save(
                os.path.join(image_dir, "all_rotations.npy"),
                poses_np[:, :3, :3],
            )
            print(f"[pipeline] All poses  saved → {image_dir}/all_poses.npy")

        # ref_idx = 0
        # ref_pose = poses_np[ref_idx]
        # merged_depth_np = self.merged_distance_map(
        #     merged_xyz, ref_pose,
        #     ref_points_hw3=world_pts_np[ref_idx],
        #     transform_path=os.path.join(image_dir, "ref_geometry", "ref_transform.npz"),
        # )

        

        print(f"\n✅ Done.  Results written to: {image_dir}")
        print("depth_np")

        return depth_np
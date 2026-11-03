cat > check_distance.py << 'EOF'
import sys, os
import numpy as np, cv2

image_dir = sys.argv[1]
d = np.squeeze(np.load(os.path.join(image_dir, "ref_distance", "distance.npy")))
print("shape", d.shape, "dtype", d.dtype)

v = d[np.isfinite(d) & (d > 0)]
print("zero/invalid ratio", 1 - v.size / d.size)
print("min", v.min(), "p1", np.percentile(v, 1), "median", np.median(v),
      "p99", np.percentile(v, 99), "max", v.max())

# 퍼센타일 정규화 + log 시각화 (outlier 에 안 끌려감)
lo, hi = np.percentile(v, [2, 98])
vis = np.clip((np.log(np.clip(d, lo, hi)) - np.log(lo)) / (np.log(hi) - np.log(lo)), 0, 1)
out = os.path.join(image_dir, "distance_check.png")
cv2.imwrite(out, cv2.applyColorMap((vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO))
print("saved", out)

# PanoVGGT 가 예측한 카메라 translation
poses = np.load(os.path.join(image_dir, "all_poses.npy"))
for i, p in enumerate(poses):
    print(f"PanoVGGT pose[{i}] t =", np.round(p[:3, 3], 4))
EOF

IMG_DIR=/data/intern01/PeRF_multi/example_data/bmw_sample_132/132_to_180
python check_distance.py $IMG_DIR
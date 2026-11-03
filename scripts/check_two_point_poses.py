"""TwoPointPoseSampler가 base_point / base_point+z_offset 두 포즈를 올바르게 주는지 확인."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from modules.pose_sampler import TwoPointPoseSampler

def main():
    base_point = [0.0, 0.0, 0.0]
    z_offset = 0.2

    sampler = TwoPointPoseSampler(base_point=base_point, z_offset=z_offset)
    assert sampler.n_poses == 2 and sampler.n_anchors == 2

    pose0 = sampler.sample_pose(0)
    pose1 = sampler.sample_pose(1)

    print("TwoPointPoseSampler 검증")
    print("  base_point:", base_point)
    print("  z_offset:", z_offset)
    print("  anchor_pts[0]:", sampler.anchor_pts[0].tolist())
    print("  anchor_pts[1]:", sampler.anchor_pts[1].tolist())
    print("  pose[0] translation:", pose0[:3, 3].tolist())
    print("  pose[1] translation:", pose1[:3, 3].tolist())
    print("  pose[1] - pose[0] (Z만 차이):", (pose1[:3, 3] - pose0[:3, 3]).tolist())

    expected_diff = [0.0, 0.0, z_offset]
    actual_diff = (pose1[:3, 3] - pose0[:3, 3]).tolist()
    assert all(abs(a - e) < 1e-5 for a, e in zip(actual_diff, expected_diff)), (
        f"두 번째 포즈는 첫 번째에서 (0,0,z_offset)만큼 떨어져야 함. actual_diff={actual_diff}"
    )
    print("  -> 두 포즈가 base_point / base_point+(0,0,z_offset) 로 올바르게 샘플링됨.")

if __name__ == "__main__":
    main()

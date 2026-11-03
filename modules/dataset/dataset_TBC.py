import os.path
import numpy as np
import torch
import trimesh
import hashlib
import cv2 as cv
import torch.nn.functional as F
import os

from os.path import join as pjoin
from utils.utils import read_dpt, read_image, write_image
from utils.camera_utils import *
from utils.geo_utils import align_scale, get_edge_mask
from modules.geo_predictors import PanoFusionInvPredictor, PanoFusionNormalPredictor, PanoGeoRefiner, PanoJointPredictor



class Dataset:
    def __init__(self):

        self.n_images = 0
        self.image_dir = None
        self.image_names = []
        self.ref_distance_path = None
        self.ref_normal_path = None
        self.ref_geometry_path = None
        self.images = None
        self.gt_distances = []
        self.ref_distances = []
        self.ref_normals = []
        self.height = 0
        self.width = 0
        self.data_dir = None
        self.poses = []
        self.case_name = 'wp'
        # self.reference_idx = 0

    def get_ref_distance(self):
        assert self.images is not None
        assert self.ref_distance_path is not None
        assert self.height > 0 and self.width > 0

        ref_distances = []
        distance_predictor = PanoFusionInvPredictor()

        zero_mask = torch.zeros([self.height, self.width])
        ones_mask = torch.ones([self.height, self.width])
       
        if os.path.exists(self.ref_distance_path):
            ref_distances = np.load(self.ref_distance_path)
            ref_distances = torch.from_numpy(ref_distances.astype(np.float32)).cuda()
            

        else:
            ref_distances, _ = distance_predictor(self.images[0], self.images[1],
                                                    zero_mask,
                                                    ones_mask)

        return ref_distances

    def get_ref_normal(self):
        assert self.images is not None
        assert self.ref_normal_path is not None
        assert self.height > 0 and self.width > 0

        ref_normals = []
        normal_predictor = PanoFusionNormalPredictor()

        norm_mask = torch.ones([self.height, self.width, 3]) / np.sqrt(3.)
        ones_mask = torch.ones([self.height, self.width])

        if os.path.exists(self.ref_normal_path):
            ref_normals = np.load(self.ref_normal_path)
            ref_normals = torch.from_numpy(ref_normals.astype(np.float32)).cuda()
        else:
            ref_normals = normal_predictor.inpaint_normal(self.images[0], self.images[1],
                                                          norm_mask,
                                                          ones_mask)

        return ref_normals

    def refine_geometry(self, distance_map, normal_map):
        refiner = PanoGeoRefiner()
        return refiner.refine(distance_map, normal_map)

    def get_joint_distance_normal(self, org_distance=None):
        assert self.images is not None
        assert self.ref_distance_path is not None
        assert self.ref_normal_path is not None
        assert self.height > 0 and self.width > 0

        ref_distances = []
        ref_normals = []

        ones_dim_mask = torch.ones([self.height, self.width, 1])
        ones_mask = torch.ones([self.height, self.width])
        zero_mask = torch.zeros([self.height, self.width])

        joint_predictor = PanoJointPredictor()

        zero_distance = torch.zeros([self.height, self.width, 1])


        # for i in range(self.n_images):

        if os.path.exists(self.ref_distance_path) and\
        os.path.exists(self.ref_normal_path):
            ref_distances = np.load(self.ref_distance_path)
            ref_distances = torch.from_numpy(ref_distances.astype(np.float32)).cuda()
            ref_normals = np.load(self.ref_normal_path)
            ref_normals = torch.from_numpy(ref_normals.astype(np.float32)).cuda()
        

        else:
            if org_distance is None:
                ref_distances, ref_normals = joint_predictor(self.images[0], self.images[1],
                                                        ref_distance1=zero_distance, 
                                                        ref_distance2=zero_distance,
                                                        mask=torch.ones([self.height, self.width]))
                
            else:
                ref_distances, ref_normals = joint_predictor(self.images[0], self.images[1],
                                                        ref_distance=org_distance,
                                                        mask=zero_mask,
                                                        reg_loss_weight=0.)

        if ref_distances.shape[0] != self.height or ref_distances.shape[1] != self.width:
            ref_distances = F.interpolate(
                ref_distances.unsqueeze(0).unsqueeze(0),
                size=(self.height, self.width),
                mode='bilinear',
                align_corners=False
            ).squeeze()

            ref_normals = F.interpolate(
                ref_normals.permute(2,0,1).unsqueeze(0),
                size=(self.height, self.width),
                mode='bilinear',
                align_corners=False
            ).squeeze().permute(1,2,0)

        return ref_distances, ref_normals

    def normalization(self):
        # 모든 뷰에 공통 scale을 적용해 뷰 간 절대 거리 단위를 일치시킴
        
        # global_max = max(d.max().item() for d in self.ref_distances)
        # scale = global_max * 1.05
      
        # self.ref_distances /= scale

        global_max = self.ref_distances.max().item()
        scale = global_max * 1.05
        self.ref_distances = self.ref_distances / scale

    def save_ref_geometry(self):
        """npy 저장, 첫 뷰 로컬 geometry.ply, 다중 뷰+pose 일치 시 월드 병합 geometry_merged.ply."""
        os.makedirs(pjoin(self.image_dir, 'ref_geometry'), exist_ok=True)
        os.makedirs(pjoin(self.image_dir, 'ref_distance'), exist_ok=True)
        os.makedirs(pjoin(self.image_dir, 'ref_normal'), exist_ok=True)

        # 단일 tensor 직접 사용
        rd = self.ref_distances  # [H, W] or [H, W, 1]
        rn = self.ref_normals    # [H, W, 3]

        np.save(self.ref_distance_path, rd.cpu().numpy())
        np.save(self.ref_normal_path,   rn.cpu().numpy())

        pano_dirs = img_coord_to_pano_direction(img_coord_from_hw(self.height, self.width))
        pts_local = pano_dirs * rd.squeeze()[..., None]

        pcd = trimesh.PointCloud(
            pts_local.cpu().numpy().reshape(-1, 3),
            vertex_colors=self.images[0].reshape(-1, 3).detach().cpu().numpy()
        )
        pcd.export(self.ref_geometry_path)

    # def normalize_poses(self):
    #     cam_positions = torch.stack([
    #         -torch.matmul(p[:3, :3].T, p[:3, 3])
    #         for p in self.poses
    #     ])  # [N, 3]
    #     global_min = cam_positions.min()   # scalar
    #     global_max = cam_positions.max()   # scalar
    #     denom = (global_max - global_min).clamp(min=1e-6)
    #     for i in range(self.n_images):
    #         cam_pos = cam_positions[i]
    #         new_cam_pos = (cam_pos - global_min) / denom * 2 - 1.0  # → [-1, 1]
    #         R = self.poses[i][:3, :3].clone()
    #         self.poses[i] = self.poses[i].clone()
    #         self.poses[i][:3, 3] = -torch.matmul(R, new_cam_pos)
    def normalize_poses(self):
        poses_c2w = self.poses  # [N, 4, 4]
        n = len(poses_c2w)

        # c2w → w2c
        poses_w2c = [p.inverse() for p in poses_c2w]

        # camera center (c2w 기준)
        cam_centers = torch.stack([
            p[:3, 3] for p in poses_c2w
        ])  # [N, 3]

        # center normalize
        center = cam_centers.mean(dim=0)
        cam_centers = cam_centers - center

        # uniform scale (중요)
        scale = cam_centers.norm(dim=1).max()
        scale = scale.clamp(min=1e-6)

        cam_centers = cam_centers / scale

        # pose 재구성 (c2w)
        new_c2w = []
        for i in range(n):
            new_pose = poses_c2w[i].clone()
            new_pose[:3, 3] = cam_centers[i]
            new_c2w.append(new_pose)

        # 다시 w2c로 변환
        # new_w2c = [p.inverse() for p in new_c2w]

        self.poses = new_c2w

    #폐기. 이상하게 렌더링 결과 출력됨.
    # def normalize_poses_simple(self):
    #     poses = torch.stack(self.poses)
    #     poses_min = poses.min()
    #     poses_max = poses.max()
    #     self.poses = (poses - poses_min) / (poses_max - poses_min) * 2 - 1.0

    def fit_scene_to_aabb(self, aabb_extent=0.9):
        max_extent = max(
            (-p[:3, :3].T @ p[:3, 3]).norm().item()
            for p in self.poses
        ) + self.ref_distances.max().item()

        if max_extent > aabb_extent:
            scale = max_extent / aabb_extent
            self.ref_distances = self.ref_distances / scale
            for i in range(self.n_images):
                new_pose = self.poses[i].clone()
                new_pose[:3, 3] = self.poses[i][:3, 3] / scale
                self.poses[i] = new_pose

class WildDataset(Dataset):
    def __init__(self, conf):
        super().__init__()
         
        self.image_dir = conf.image_dir

        # png 파일 자동 탐색
        self.image_names = sorted([
            os.path.splitext(f)[0]   # 확장자 제거
            for f in os.listdir(self.image_dir)
            if f.endswith('.jpg')
        ])
        self.n_images = len(self.image_names)

        #image_dir + image_names[i] -> 이미지 저장 경로
        self.ref_distance_path = pjoin(self.image_dir, 'ref_distance', f"distance.npy") 
        self.ref_normal_path = pjoin(self.image_dir, 'ref_normal', f"normal.npy") 
        self.ref_geometry_path = pjoin(self.image_dir, 'ref_geometry', f"geometry.ply") 
        self.warped_image_dir = pjoin(self.image_dir, 'warped_images')
        # self.ref_distance_path = '.'.join(self.image_path.split('.')[:-1]) + '_ref_distance.npy'
        # self.ref_normal_path = '.'.join(self.image_path.split('.')[:-1]) + '_ref_normal.npy'
        # self.ref_geometry_path = '.'.join(self.image_path.split('.')[:-1]) + '_ref_geometry.ply'

        # self.case_name = self.image_path.split('/')[-2]
        self.images = [read_image(pjoin(self.image_dir, name + ".jpg"), to_torch=True, squeeze=True).cuda() for name in self.image_names]

        self.case_name = os.path.basename(self.image_dir)

        if 'image_resize' in conf:
            self.width, self.height = conf['image_resize']
            for i in range(self.n_images):
                self.images[i] = cv.resize(self.images[i].cpu().numpy(), (self.width, self.height), cv.INTER_AREA)
                self.images[i] = torch.from_numpy(self.images[i]).cuda()
        else:
            shapes = [img.shape for img in self.images]
            assert all(s == shapes[0] for s in shapes), "Shapes must be equal for all images"
            self.height, self.width, _ = self.images[0].shape

        self.ref_distances, self.ref_normals = self.get_joint_distance_normal()
        
        self.poses = load_cam_extrinsic(self.image_dir) #camera coordinate(w2c)

        # self.reference_idx = 0
        # self.images = self.reprojection_to_reference(self.reference_idx)
        # self.save_warped_images(self.warped_image_dir)

        # self.normalization()
        # self.normalize_poses_simple()
        # self.normalize_poses()
        # self.fit_scene_to_aabb()

        self.save_ref_geometry()


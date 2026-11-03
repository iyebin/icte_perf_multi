import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
from tqdm import tqdm
from kornia.morphology import erosion, dilation
import os

from .inpainter import Inpainter
from .lama_inpainter import LamaInpainter
from .diffusion_local_inpainter import DiffusionLocalInpainter

from utils.geo_utils import panorama_to_pers_directions
from utils.camera_utils import img_coord_to_sample_coord,\
    direction_to_img_coord, img_coord_to_pano_direction, direction_to_pers_img_coord

from utils.utils import write_image


def _save_rgb_float_hwc(path, hwc_rgb):
    """Debug dump; match core_exp_runner: write_image(path, rgb * 255.) on a float tensor."""
    if torch.is_tensor(hwc_rgb):
        img = hwc_rgb.detach().float()
    else:
        img = torch.as_tensor(np.asarray(hwc_rgb), dtype=torch.float32, device='cpu')
    write_image(path, img * 255.0)


class PanoPersFusionInpainter(Inpainter):
    def __init__(self, inpainter_type, use_lama_assist=True):
        super().__init__()
        if inpainter_type == 'stable_diffusion':
            self.diff_inpainter = DiffusionLocalInpainter()
        else:
            assert False
        if use_lama_assist:
            self.lama_inpainter = LamaInpainter()
        else:
            self.lama_inpainter = None

    @torch.no_grad()
    def inpaint(self, img, mask, exp_dir, anchor_idx=0, phase=0):
        output_path = os.path.join(exp_dir, 'from_pano_pers_fusion_inpainter', f'{anchor_idx:02d}', f'{phase:02d}')

        os.makedirs(output_path, exist_ok=True)

        #📌1.  img: (H, W, C) 출력
        _save_rgb_float_hwc(os.path.join(output_path, f'img(before)_{anchor_idx:02d}_{phase:02d}.png'), img.squeeze())    
        #📌1.  mask: (H, W) 출력
        output_mask = mask.squeeze()
        output_mask = output_mask.unsqueeze(-1).repeat(1, 1, 3)
        _save_rgb_float_hwc(os.path.join(output_path, f'mask(before)_{anchor_idx:02d}_{phase:02d}.png'), output_mask)

        img = img.squeeze().permute(2, 0, 1) #(H, W, C) -> (C, H, W)
        mask = mask.squeeze()[None] #(H, W) -> (1, H, W)
        inpainted_img = img.clone() #결과 누적

        pers_dirs, pers_ratios, to_vecs, down_vecs, right_vecs = panorama_to_pers_directions(gen_res=512, ratio=1.1)
        pers_dirs = pers_dirs.to(device=img.device, dtype=img.dtype)
        to_vecs = to_vecs.to(device=img.device, dtype=img.dtype)
        down_vecs = down_vecs.to(device=img.device, dtype=img.dtype)
        right_vecs = right_vecs.to(device=img.device, dtype=img.dtype)

        n_pers = len(pers_dirs) 
        img_coords = direction_to_img_coord(pers_dirs) #방향벡터 -> 이미지 좌표
        sample_coords = img_coord_to_sample_coord(img_coords)

        _, pano_height, pano_width = img.shape
        pano_img_coords = torch.meshgrid(
            torch.linspace(.5 / pano_height, 1. - .5 / pano_height, pano_height, device=img.device, dtype=img.dtype),
            torch.linspace(.5 / pano_width, 1. - .5 / pano_width, pano_width, device=img.device, dtype=img.dtype),
            indexing='ij')
        pano_img_coords = torch.stack(list(pano_img_coords), dim=-1)

        pano_dirs = img_coord_to_pano_direction(pano_img_coords) #파노라마의 모든 픽셀을 방향 벡터로 변환

        #⭐핵심
        for i in tqdm(range(n_pers)):
            cur_sample_coords = sample_coords[i]
            
            
            pers_image = F.grid_sample(inpainted_img[None], cur_sample_coords[None], padding_mode='border')[0]
            #2. 📌pers_image: 현재 픽셀의 이미지 값 출력
            output_pers_image = pers_image.clone()
            output_pers_image = output_pers_image.permute(1, 2, 0)
            _save_rgb_float_hwc(os.path.join(output_path, f'pers_image_{i}_{anchor_idx:02d}_{phase:02d}.png'), output_pers_image)

            pers_mask = F.grid_sample(mask[None, :, :], cur_sample_coords[None], padding_mode='border')[0]
            pers_mask = (pers_mask > 0.5).float() #inpainting 영역만 남김
            #2. 📌pers_mask: 현재 픽셀의 마스크 값 출력
            output_pers_mask = pers_mask.squeeze()
            output_pers_mask = output_pers_mask.unsqueeze(-1).repeat(1, 1, 3)
            _save_rgb_float_hwc(os.path.join(output_path, f'pers_mask(before)_{i}_{anchor_idx:02d}_{phase:02d}.png'), output_pers_mask)
            
            if self.lama_inpainter is not None:
                #LaMa + Stable Diffusion
                kernel = torch.from_numpy(cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))).float().to(pers_mask.device)
                smooth_mask = erosion(pers_mask[None], kernel=kernel)[0]
                smooth_mask = dilation(smooth_mask[None], kernel=kernel)[0] 
                smooth_mask = torch.minimum(smooth_mask, pers_mask) #경계 부드럽게
                
                lama_inpainted = self.lama_inpainter.inpaint(pers_image[None], pers_mask[None])[0] #1차 inpainting(LaMa)
                #3. 📌 lama_inpainted
                output_lama_inpainted = lama_inpainted.clone()
                output_lama_inpainted = output_lama_inpainted.permute(1, 2, 0)
                _save_rgb_float_hwc(os.path.join(output_path, f'lama_inpainted_{i}_{anchor_idx:02d}_{phase:02d}.png'), output_lama_inpainted)
                
                if smooth_mask.max().item() > .5: #2차 inpainting (Stable Diffusion) 필요 시 적용
                    cur_inpainted = self.diff_inpainter.inpaint(lama_inpainted[None], smooth_mask[None])[0] #2차 inpainting(Stable Diffusion)
                    #4. 📌 cur_inpainted
                    output_cur_inpainted = cur_inpainted.clone()
                    output_cur_inpainted = output_cur_inpainted.permute(1, 2, 0)
                    _save_rgb_float_hwc(os.path.join(output_path, f'diffusion_inp   ainted_{i}_{anchor_idx:02d}_{phase:02d}.png'), output_cur_inpainted)
                else:
                    cur_inpainted = lama_inpainted
            else:
                if pers_mask.max().item() > .5:
                    cur_inpainted = self.diff_inpainter.inpaint(pers_image[None], pers_mask[None])[0]
                else:
                    cur_inpainted = pers_image

            #panorama로 다시 투영
            proj_coord, proj_mask = direction_to_pers_img_coord(pano_dirs, to_vecs[i], down_vecs[i], right_vecs[i])
            proj_coord = img_coord_to_sample_coord(proj_coord)

            cur_inpainted_pano_img = F.grid_sample(cur_inpainted[None], proj_coord[None], padding_mode='border')[0]
            proj_mask = proj_mask.permute(2, 0, 1).float()
            # 최종 결과물
            inpainted_img = inpainted_img * (1. - proj_mask) + cur_inpainted_pano_img * proj_mask

            #5. 📌 final_inpainted_img
            final_inpainted_img = inpainted_img.clone()
            final_inpainted_img = final_inpainted_img.permute(1, 2, 0)
            _save_rgb_float_hwc(os.path.join(output_path, f'final_inpainted_{i}_{anchor_idx:02d}_{phase:02d}.png'), final_inpainted_img)
        
            #mask 업데이트
            mask = mask * (1. - proj_mask) + 0. * proj_mask

        return inpainted_img.permute(1, 2, 0) 


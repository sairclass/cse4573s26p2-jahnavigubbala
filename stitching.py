'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function stitch_background() and panorama().
3. If you want to show an image for debugging, please use show_image() function in util.py. 
4. Please do NOT save any intermediate files in your final submission.
'''
import torch
import kornia as K
from typing import Dict
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

# ------------------------------------ Task 1 ------------------------------------ #
def stitch_background(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: input images are a dict of 2 images of torch.Tensor represent an input images for task-1.
    Returns:
        img: stitched_image: torch.Tensor of the output image.
    """
    img = torch.zeros((3, 256, 256)) # assumed 256*256 resolution. Update this as per your logic.

    #TODO: Add your code here. Do not modify the return and input arguments.
    names = sorted(imgs.keys())
    img1 = imgs[names[0]].float() / 255.0
    img2 = imgs[names[1]].float() / 255.0

    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)

    gray1 = K.color.rgb_to_grayscale(img1)
    gray2 = K.color.rgb_to_grayscale(img2)

    sift = K.feature.SIFTFeature(num_features=800)

    lafs1, resp1, desc1 = sift(gray1)
    lafs2, resp2, desc2 = sift(gray2)

    pts1 = K.feature.get_laf_center(lafs1).reshape(-1,2)
    pts2 = K.feature.get_laf_center(lafs2).reshape(-1,2)

    desc1 = desc1.reshape(-1, desc1.shape[-1])
    desc2 = desc2.reshape(-1, desc2.shape[-1])

    dists, idxs = K.feature.match_smnn(desc1, desc2, 0.8)

    if idxs.shape[0] < 4:
        return (img1[0]*255).byte()

    src = pts1[idxs[:,0]]
    dst = pts2[idxs[:,1]]
    
    best_H = None
    best_inliers = None
    best_count = 0
    thresh = 4.0
    num_matches = src.shape[0]

    if num_matches < 4:
        return (img1[0] * 255).byte()

    for _ in range(1000):
        perm = torch.randperm(num_matches)[:4]
        src4 = src[perm].unsqueeze(0)   # 1x4x2
        dst4 = dst[perm].unsqueeze(0)   # 1x4x2

        try:
            Hcand = K.geometry.find_homography_dlt(src4, dst4)[0]   # 3x3
        except:
            continue

        src_h = torch.cat([src, torch.ones(num_matches, 1)], dim=1)   # Nx3
        proj = (Hcand @ src_h.T).T
        proj = proj[:, :2] / (proj[:, 2:3] + 1e-8)

        err = torch.norm(proj - dst, dim=1)
        inliers = err < thresh
        count = int(inliers.sum().item())

        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_H = Hcand

    if best_H is None or best_count < 4:
        return (img1[0] * 255).byte()

    H = K.geometry.find_homography_dlt(
        src[best_inliers].unsqueeze(0),
        dst[best_inliers].unsqueeze(0)
    )[0]

    _,_,h1,w1 = img1.shape
    _,_,h2,w2 = img2.shape

    corners1 = torch.tensor([[0,0],[w1,0],[w1,h1],[0,h1]], dtype=torch.float32)
    corners2 = torch.tensor([[0,0],[w2,0],[w2,h2],[0,h2]], dtype=torch.float32)

    corners2_warp = K.geometry.transform_points(H.unsqueeze(0), corners2.unsqueeze(0))[0]

    all_x = torch.cat([corners1[:,0], corners2_warp[:,0]])
    all_y = torch.cat([corners1[:,1], corners2_warp[:,1]])

    min_x = int(torch.floor(all_x.min()).item())
    min_y = int(torch.floor(all_y.min()).item())
    max_x = int(torch.ceil(all_x.max()).item())
    max_y = int(torch.ceil(all_y.max()).item())

    out_w = max_x - min_x
    out_h = max_y - min_y

    T = torch.tensor([
        [1,0,-min_x],
        [0,1,-min_y],
        [0,0,1]
    ], dtype=torch.float32)

    H2 = T @ H
    H1 = T

    warp1 = K.geometry.warp_perspective(img1, H1.unsqueeze(0), (out_h, out_w))
    warp2 = K.geometry.warp_perspective(img2, H2.unsqueeze(0), (out_h, out_w))

    mask1 = (warp1>0)
    mask2 = (warp2>0)

    both = mask1 & mask2
    only1 = mask1 & (~mask2)
    only2 = mask2 & (~mask1)

    out = torch.zeros_like(warp1)

    out += warp1 * only1
    out += warp2 * only2
    out += (warp1+warp2)/2 * both

    out = torch.clamp(out[0]*255,0,255).byte()

    img = out
    return img

# ------------------------------------ Task 2 ------------------------------------ #
def panorama(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: dict {filename: CxHxW tensor} for task-2.
    Returns:
        img: panorama, 
        overlap: torch.Tensor of the output image. 
    """
    img = torch.zeros((3, 256, 256)) # assumed 256*256 resolution. Update this as per your logic.
    overlap = torch.empty((3, 256, 256)) # assumed empty 256*256 overlap. Update this as per your logic.

    #TODO: Add your code here. Do not modify the return and input arguments.

    return img, overlap

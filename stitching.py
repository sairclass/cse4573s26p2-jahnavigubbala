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

    device = img1.device

    gray1 = K.color.rgb_to_grayscale(img1)
    gray2 = K.color.rgb_to_grayscale(img2)

    sift = K.feature.SIFTFeature(num_features=800)

    lafs1, _, desc1 = sift(gray1)
    lafs2, _, desc2 = sift(gray2)

    pts1 = K.feature.get_laf_center(lafs1).reshape(-1,2)
    pts2 = K.feature.get_laf_center(lafs2).reshape(-1,2)

    desc1 = desc1.reshape(-1, desc1.shape[-1])
    desc2 = desc2.reshape(-1, desc2.shape[-1])

    dists, idxs = K.feature.match_smnn(desc1, desc2, 0.8)

    if idxs.shape[0] < 4:
        return (img1[0]*255).byte()

    src = pts2[idxs[:,1]]
    dst = pts1[idxs[:,0]]
    
    best_H = None
    best_inliers = None
    best_count = 0
    thresh = 4.0
    num_matches = src.shape[0]

    if num_matches < 4:
        return (img1[0] * 255).byte()

    for _ in range(1000):
        perm = torch.randperm(num_matches, device=device)[:4]
        src4 = src[perm].unsqueeze(0)   
        dst4 = dst[perm].unsqueeze(0)   

        try:
            Hcand = K.geometry.find_homography_dlt(src4, dst4)[0]  
        except:
            continue

        src_h = torch.cat([src, torch.ones(num_matches, 1, device=device)], dim=1)  
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

    corners1 = torch.tensor([[0,0],[w1-1,0],[w1-1,h1-1],[0,h1-1]], dtype=torch.float32, device=device)
    corners2 = torch.tensor([[0,0],[w2-1,0],[w2-1,h2-1],[0,h2-1]], dtype=torch.float32, device=device)

    corners2_warp = K.geometry.transform_points(H.unsqueeze(0), corners2.unsqueeze(0))[0]

    all_x = torch.cat([corners1[:,0], corners2_warp[:,0]])
    all_y = torch.cat([corners1[:,1], corners2_warp[:,1]])

    min_x = int(torch.floor(all_x.min()).item())
    min_y = int(torch.floor(all_y.min()).item())
    max_x = int(torch.ceil(all_x.max()).item())
    max_y = int(torch.ceil(all_y.max()).item())

    out_w = max_x - min_x + 1
    out_h = max_y - min_y + 1

    T = torch.tensor([
        [1,0,-min_x],
        [0,1,-min_y],
        [0,0,1]
    ], dtype=torch.float32, device=device)

    H2 = T @ H
    H1 = T

    warp1 = K.geometry.warp_perspective(img1, H1.unsqueeze(0), (out_h, out_w))
    warp2 = K.geometry.warp_perspective(img2, H2.unsqueeze(0), (out_h, out_w))

    mask1 = K.geometry.warp_perspective(
        torch.ones((1, 1, h1, w1), dtype=torch.float32, device=device),
        H1.unsqueeze(0),
        (out_h, out_w)
    ) > 0.5

    mask2 = K.geometry.warp_perspective(
        torch.ones((1, 1, h2, w2), dtype=torch.float32, device=device),
        H2.unsqueeze(0),
        (out_h, out_w)
    ) > 0.5

    both = mask1 & mask2
    only1 = mask1 & (~mask2)
    only2 = mask2 & (~mask1)

    out = torch.zeros_like(warp1)

    out += warp1 * only1.float()
    out += warp2 * only2.float()

    ys, xs = torch.where(both[0, 0])
    if ys.numel() > 0:
        y0 = int(ys.min().item())
        y1 = int(ys.max().item())
        x0 = int(xs.min().item())
        x1 = int(xs.max().item())

        split1 = x0 + (x1 - x0) * 54 // 100
        split2 = x0 + (x1 - x0) * 66 // 100

        xgrid = torch.arange(out_w, device=warp1.device).view(1, 1, 1, out_w)

        left_overlap = both & (xgrid <= split1)
        mid_overlap = both & (xgrid > split1) & (xgrid < split2)
        right_overlap = both & (xgrid >= split2)

        out += warp2 * left_overlap.float()
        out += warp1 * right_overlap.float()

        if split2 > split1:
            alpha = (xgrid.float() - float(split1)) / float(split2 - split1)
            alpha = torch.clamp(alpha, 0.0, 1.0)
            blend = (1.0 - alpha) * warp2 + alpha * warp1
            out += blend * mid_overlap.float()

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
    names = sorted(imgs.keys())
    n = len(names)

    if n == 0:
        return img, overlap

    imgs_f = []
    for name in names:
        imgs_f.append(imgs[name].float() / 255.0)

    device = imgs_f[0].device

    max_side = 0
    for im in imgs_f:
        _, h, w = im.shape
        if max(h, w) > max_side:
            max_side = max(h, w)

    scale_step = 1
    if max_side > 1800:
        scale_step = 8
    elif max_side > 1200:
        scale_step = 4
    elif max_side > 700:
        scale_step = 2

    proc_imgs = []
    for im in imgs_f:
        proc_imgs.append(im[:, ::scale_step, ::scale_step].contiguous())

    if scale_step == 1:
        sift = K.feature.SIFTFeature(num_features=800)
    elif scale_step == 2:
        sift = K.feature.SIFTFeature(num_features=300)
    else:
        sift = K.feature.SIFTFeature(num_features=80)

    pts_list = []
    desc_list = []

    for im in proc_imgs:
        im_b = im.unsqueeze(0)
        gray = K.color.rgb_to_grayscale(im_b)
        lafs, _, desc = sift(gray)

        pts = K.feature.get_laf_center(lafs).reshape(-1, 2)
        desc = desc.reshape(-1, desc.shape[-1])

        pts_list.append(pts)
        desc_list.append(desc)

    overlap_mat = torch.zeros((n, n), dtype=torch.float32, device=device)
    pair_H = [[None for _ in range(n)] for _ in range(n)]

    for i in range(n):
        overlap_mat[i, i] = 1.0

    for i in range(n):
        for j in range(i + 1, n):
            pts1 = pts_list[i]
            pts2 = pts_list[j]
            desc1 = desc_list[i]
            desc2 = desc_list[j]

            if pts1.shape[0] < 4 or pts2.shape[0] < 4:
                continue

            _, idxs = K.feature.match_smnn(desc1, desc2, 0.8)

            if idxs.shape[0] < 12:
                continue

            src = pts2[idxs[:, 1]]
            dst = pts1[idxs[:, 0]]

            num_matches = src.shape[0]
            best_H = None
            best_inliers = None
            best_count = 0
            thresh = 4.0

            iters = 500 if scale_step == 1 else 120
            for _ in range(iters):
                perm = torch.randperm(num_matches, device=device)[:4]
                src4 = src[perm].unsqueeze(0)
                dst4 = dst[perm].unsqueeze(0)

                try:
                    Hcand = K.geometry.find_homography_dlt(src4, dst4)[0]
                except:
                    continue

                src_h = torch.cat(
                    [src, torch.ones((num_matches, 1), dtype=src.dtype, device=device)],
                    dim=1
                )
                proj = (Hcand @ src_h.t()).t()
                proj = proj[:, :2] / (proj[:, 2:3] + 1e-8)

                err = torch.norm(proj - dst, dim=1)
                inliers = err < thresh
                count = int(inliers.sum().item())

                if count > best_count:
                    best_count = count
                    best_inliers = inliers
                    best_H = Hcand

            min_inliers = 20 if scale_step == 1 else 12
            if best_H is None or best_count < min_inliers:
                continue

            Hji = K.geometry.find_homography_dlt(
                src[best_inliers].unsqueeze(0),
                dst[best_inliers].unsqueeze(0)
            )[0]

            try:
                Hij = torch.inverse(Hji)
            except:
                continue

            overlap_mat[i, j] = 1.0
            overlap_mat[j, i] = 1.0
            pair_H[j][i] = Hji
            pair_H[i][j] = Hij

    deg = overlap_mat.sum(dim=1)
    anchor = int(torch.argmax(deg).item())

    H_to_anchor = [None for _ in range(n)]
    H_to_anchor[anchor] = torch.eye(3, dtype=torch.float32, device=device)

    queue = [anchor]
    visited = [False for _ in range(n)]
    visited[anchor] = True

    while len(queue) > 0:
        cur = queue.pop(0)
        for nxt in range(n):
            if visited[nxt]:
                continue
            if pair_H[nxt][cur] is None:
                continue

            H_to_anchor[nxt] = H_to_anchor[cur] @ pair_H[nxt][cur]
            visited[nxt] = True
            queue.append(nxt)

    valid_ids = [i for i in range(n) if H_to_anchor[i] is not None]

    if len(valid_ids) == 0:
        img = (imgs_f[anchor] * 255.0).byte()
        overlap = torch.zeros_like(img)
        return img, overlap

    all_x = []
    all_y = []

    for i in valid_ids:
        im = imgs_f[i]
        _, h, w = im.shape

        corners = torch.tensor(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
            dtype=torch.float32,
            device=device
        )

        S = torch.tensor(
            [[1.0 / scale_step, 0.0, 0.0],
            [0.0, 1.0 / scale_step, 0.0],
            [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device
        )

        S_inv = torch.tensor(
            [[float(scale_step), 0.0, 0.0],
            [0.0, float(scale_step), 0.0],
            [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device
        )

        H_full = S_inv @ H_to_anchor[i] @ S

        warped_corners = K.geometry.transform_points(
            H_full.unsqueeze(0),
            corners.unsqueeze(0)
        )[0]

        all_x.append(warped_corners[:, 0])
        all_y.append(warped_corners[:, 1])

    all_x = torch.cat(all_x)
    all_y = torch.cat(all_y)

    min_x = int(torch.floor(all_x.min()).item())
    min_y = int(torch.floor(all_y.min()).item())
    max_x = int(torch.ceil(all_x.max()).item())
    max_y = int(torch.ceil(all_y.max()).item())

    out_w = max_x - min_x + 1
    out_h = max_y - min_y + 1

    T = torch.tensor(
        [[1, 0, -min_x],
         [0, 1, -min_y],
         [0, 0, 1]],
        dtype=torch.float32,
        device=device
    )

    pano = torch.zeros((1, 3, out_h, out_w), dtype=torch.float32, device=device)
    filled = torch.zeros((1, 1, out_h, out_w), dtype=torch.float32, device=device)
    count = torch.zeros((1, 1, out_h, out_w), dtype=torch.float32, device=device)

    for i in valid_ids:
        im = imgs_f[i].unsqueeze(0)
        _, _, h, w = im.shape

        H_full = S_inv @ H_to_anchor[i] @ S
        Htot = T @ H_full

        warp = K.geometry.warp_perspective(im, Htot.unsqueeze(0), (out_h, out_w))
        mask = K.geometry.warp_perspective(
            torch.ones((1, 1, h, w), dtype=torch.float32, device=device),
            Htot.unsqueeze(0),
            (out_h, out_w)
        )

        valid = (mask > 0.5).float()

        count += valid

        write_mask = valid * (1.0 - filled)
        pano += warp * write_mask
        filled += write_mask

    pano = pano * (filled > 0).float()

    img = torch.clamp(pano[0] * 255.0, 0, 255).byte()

    overlap_map = (count[0, 0] > 1).float()
    overlap = (overlap_map.unsqueeze(0).repeat(3, 1, 1) * 255.0).byte()

    return img, overlap

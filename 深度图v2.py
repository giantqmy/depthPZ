import os
import cv2
import torch
import numpy as np

from glob import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import torch.nn.functional as F


# =====================================================
# 0. 参数
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 是否保存可视化
SAVE_VIS = False


img_dir = r"I:\KEDADATA\data\night\RGB"
bin_dir = r"I:\KEDADATA\data\night\pcluse"

depth_out_dir = r"E:\depth"
depth_vis_dir = r"E:\depth_vis_gray"

os.makedirs(
    depth_vis_dir,
    exist_ok=True
)
vis_out_dir = r"E:\depth_vis"


os.makedirs(depth_out_dir, exist_ok=True)

if SAVE_VIS:
    os.makedirs(vis_out_dir, exist_ok=True)



# =====================================================
# 1. 标定参数
# =====================================================


P2_np = np.array([
    [1265.5616099448073, 0, 492.5894472471048],
    [0,1265.0089455071945,257.24307296897143],
    [0,0,1]
],dtype=np.float32)


Tr_np = np.array([
    [-0.0147331,-0.999842,-0.00999829,-0.0153519],
    [0.00641986,0.00990464,-0.99993,0.057766],
    [0.999871,-0.0147963,0.00627293,0.12535]
],dtype=np.float32)



# GPU tensor

P2 = torch.tensor(
    P2_np,
    device=device
)


Tr = torch.tensor(
    Tr_np,
    device=device
)



# =====================================================
# 2. 数据读取
# =====================================================


def load_sample(args):

    img_path, bin_path = args


    img=cv2.imread(
        img_path
    )


    points=np.fromfile(
        bin_path,
        dtype=np.float32
    ).reshape(-1,4)


    points=points[:,:3]


    return (
        os.path.basename(img_path),
        img,
        points
    )



# =====================================================
# 3. GPU LiDAR投影
# =====================================================


@torch.no_grad()
def lidar_projection_gpu(points,img_shape):


    h,w=img_shape[:2]


    pts=torch.from_numpy(
        points
    ).float().to(device)



    ones=torch.ones(
        (pts.shape[0],1),
        device=device
    )


    pts_h=torch.cat(
        [
            pts,
            ones
        ],
        dim=1
    )


    # lidar -> camera

    pts_cam=torch.matmul(
        pts_h,
        Tr.T
    )


    mask=pts_cam[:,2]>0


    pts_cam=pts_cam[mask]



    # camera -> image


    proj=torch.matmul(
        pts_cam,
        P2.T
    )


    u=(
        proj[:,0]/
        proj[:,2]
    ).long()


    v=(
        proj[:,1]/
        proj[:,2]
    ).long()



    depth=pts_cam[:,2]



    valid=(
        (u>=0)&
        (u<w)&
        (v>=0)&
        (v<h)
    )


    u=u[valid]
    v=v[valid]
    depth=depth[valid]



    # sparse depth

    index=v*w+u



    depth_flat=torch.full(
        (h*w,),
        float("inf"),
        device=device
    )


    depth_flat.scatter_reduce_(
        0,
        index,
        depth,
        reduce="amin",
        include_self=True
    )


    depth_flat[
        torch.isinf(depth_flat)
    ]=0



    return depth_flat.reshape(
        h,w
    )



# =====================================================
# 4. GPU densify
# =====================================================


@torch.no_grad()
def densify_gpu(
        depth,
        kernel_size=3,
        max_diff=0.3):


    x=depth[
        None,
        None
    ]


    patches=F.unfold(
        x,
        kernel_size=kernel_size,
        padding=kernel_size//2
    )


    patches=patches[0]


    valid=patches>0



    values=torch.where(
        valid,
        patches,
        torch.nan
    )


    median=torch.nanmedian(
        values,
        dim=0
    ).values



    diff=torch.abs(
        patches-median
    )


    close=(
        valid &
        (diff<max_diff)
    )



    fill=torch.where(
        close,
        patches,
        torch.nan
    )



    new_depth=torch.nanmedian(
        fill,
        dim=0
    ).values



    center=depth.reshape(-1)


    mask=(
        (center==0)&
        (~torch.isnan(new_depth))
    )


    center[mask]=new_depth[mask]



    return center.reshape(
        depth.shape
    )



# =====================================================
# 5. 可视化函数
# =====================================================


def visualize_projection(
        img,
        points,
        save_path):


    h,w=img.shape[:2]


    pts_h=np.hstack(
        [
            points,
            np.ones(
                (points.shape[0],1)
            )
        ]
    )


    pts_cam=(
        Tr_np @ pts_h.T
    ).T


    mask=pts_cam[:,2]>0

    pts_cam=pts_cam[mask]


    proj=(
        P2_np @ pts_cam[:,:3].T
    ).T



    u=(
        proj[:,0]/
        proj[:,2]
    ).astype(np.int32)


    v=(
        proj[:,1]/
        proj[:,2]
    ).astype(np.int32)



    depth=pts_cam[:,2]


    valid=(
        (u>=0)&
        (u<w)&
        (v>=0)&
        (v<h)
    )


    u=u[valid]
    v=v[valid]
    depth=depth[valid]


    vis=img.copy()



    norm=cv2.normalize(
        depth,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)


    colors=cv2.applyColorMap(
        norm,
        cv2.COLORMAP_JET
    )
    colors=colors.reshape(-1,3)




    for x,y,c in zip(u,v,colors):

        cv2.circle(
            vis,
            (int(x),int(y)),
            2,
            c.tolist(),
            -1
        )


    cv2.imwrite(
        save_path,
        vis
    )




def depth_visualize(
        depth,
        save_path):


    depth=depth.copy()


    mask=depth>0


    vis=np.zeros(
        depth.shape,
        dtype=np.uint8
    )


    if mask.any():

        d=depth[mask]


        d_norm=(
            (d-d.min())/
            (d.max()-d.min()+1e-6)
            *255
        )


        vis[mask]=d_norm.astype(
            np.uint8
        )


    vis=cv2.applyColorMap(
        vis,
        cv2.COLORMAP_JET
    )


    cv2.imwrite(
        save_path,
        vis
    )



# =====================================================
# 6. 主程序
# =====================================================


if __name__=="__main__":


    img_files=sorted(
        glob(
            os.path.join(
                img_dir,
                "*.png"
            )
        )
    )



    samples=[]


    for img_path in img_files:


        name=os.path.splitext(
            os.path.basename(img_path)
        )[0]


        bin_path=os.path.join(
            bin_dir,
            name+".bin"
        )


        if os.path.exists(bin_path):

            samples.append(
                (
                    img_path,
                    bin_path
                )
            )


    print(
        "Total samples:",
        len(samples)
    )



    with ThreadPoolExecutor(
        max_workers=8
    ) as loader:



        for name,img,points in tqdm(
            loader.map(
                load_sample,
                samples
            ),
            total=len(samples)
        ):



            # -----------------
            # GPU sparse depth
            # -----------------

            sparse=lidar_projection_gpu(
                points,
                img.shape
            )



            # -----------------
            # GPU densify
            # -----------------

            dense=densify_gpu(
                sparse,
                kernel_size=5,
                max_diff=0.5
            )



            # -----------------
            # 保存16bit depth
            # -----------------

            dense_np=dense.cpu().numpy()


            depth_u16=np.clip(
                dense_np*256,
                0,
                65535
            ).astype(
                np.uint16
            )


            cv2.imwrite(
                os.path.join(
                    depth_out_dir,
                    name.replace(
                        ".png",
                        "_depth_dense_u16.png"
                    )
                ),
                depth_u16
            )

            mask = dense_np > 0

            depth_vis = np.zeros(
                dense_np.shape,
                dtype=np.uint8
            )


            if mask.any():

                depth_valid = dense_np[mask]

                depth_norm = (
                    (depth_valid - depth_valid.min())
                    /
                    (depth_valid.max() - depth_valid.min() + 1e-6)
                    * 255
                )


                depth_vis[mask] = depth_norm.astype(
                    np.uint8
                )


            cv2.imwrite(
                os.path.join(
                    depth_vis_dir,
                    name.replace(
                        ".png",
                        "_depth_vis.png"
                    )
                ),
                depth_vis
            )

            # -----------------
            # 可视化
            # -----------------

            if SAVE_VIS:


                visualize_projection(
                    img,
                    points,
                    os.path.join(
                        vis_out_dir,
                        name.replace(
                            ".png",
                            "_lidar_proj.png"
                        )
                    )
                )


                depth_visualize(
                    sparse.cpu().numpy(),
                    os.path.join(
                        vis_out_dir,
                        name.replace(
                            ".png",
                            "_sparse.png"
                        )
                    )
                )


                depth_visualize(
                    dense_np,
                    os.path.join(
                        vis_out_dir,
                        name.replace(
                            ".png",
                            "_dense.png"
                        )
                    )
                )



            print(
                "✔",
                name
            )

import os
import pickle
import json
from glob import glob
from datetime import datetime as dt
from scipy.spatial.transform import Rotation as R
import argparse

import numpy as np
from tqdm import tqdm

try:
    import cv2
except ImportError:
    cv2 = None

# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------

CALIBRATION_FILES = {
    "ar0820": "calibrations/germany/calibration/calibrations_rccb.json",  # RCCB
    "rgb": "calibrations/germany/calibration/calibration_cam_stereo.json",  # cam_stereo
    "gated": "calibrations/germany/calibration/calibration_gated.json",
}

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)

def quat_conj_xyzw(q):
    x, y, z, w = q
    return np.array([-x, -y, -z, w], dtype=float)

def quat_to_rotmat_xyzw(q):
    # normalized
    x, y, z, w = quat_normalize(q)

    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    return np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ])

def invert_qt_xyzw(q, t):
    q = np.asarray(q, dtype=float)
    t = np.asarray(t, dtype=float)

    q_inv = quat_conj_xyzw(q)                # inverse quaternion
    R_inv = quat_to_rotmat_xyzw(q_inv)       # rotation matrix of inverse
    t_inv = -(R_inv @ t)                     # inverse translation
    return q_inv, t_inv


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def rot_to_quat(R: np.ndarray):
    """Rotation matrix -> quaternion [x, y, z, w]."""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    tr = m00 + m11 + m22
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S

    return [float(x), float(y), float(z), float(w)]

def quat_to_rot(qx, qy, qz, qw):
    """Quaternion -> 3x3 rotation matrix (x, y, z, w)."""
    x, y, z, w = qx, qy, qz, qw
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n == 0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    return R

def list_quat_to_T(t_list, q_list):
    """t_list: [tx,ty,tz], q_list: [qx,qy,qz,qw] -> 4x4 matrix."""
    if t_list is None or q_list is None:
        return None
    Rm = quat_to_rot(q_list[0], q_list[1], q_list[2], q_list[3])
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3] = t_list
    return T


# --- FAST-ish image shape (header-only via PIL when available) ---
def get_image_shape_header(img_path):
    """
    Returns (H, W, C) or None if cannot read.
    Attempts header-only read via PIL (fast) and falls back to cv2.imread if needed.
    """
    if img_path is None:
        return None
    try:
        from PIL import Image
        with Image.open(img_path) as im:
            w, h = im.size
            c = len(im.getbands())
        return (int(h), int(w), int(c))
    except Exception:
        # fallback (slower)
        if cv2 is None:
            return None
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 2:
            h, w = img.shape
            c = 1
        else:
            h, w, c = img.shape
        return (int(h), int(w), int(c))


def get_image_shape(img_path):
    """
    Returns (H, W, C) or None if cannot read.
    NOTE: This decodes the image when using cv2. Prefer get_image_shape_header()
    when you only need shape.
    """
    if img_path is None:
        return None

    if cv2 is not None:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 2:
            h, w = img.shape
            c = 1
        else:
            h, w, c = img.shape
        return (int(h), int(w), int(c))
    else:
        from PIL import Image
        with Image.open(img_path) as im:
            w, h = im.size
            c = len(im.getbands())
        return (int(h), int(w), int(c))


# --- timestamp helpers (avoid datetime + O(N^2) matching) ---
def ts_ns_from_path(p):
    # ..._<nanoseconds>.png/.npz/.bin
    base = os.path.basename(p)
    stem, _ = os.path.splitext(base)
    return int(stem.split("_")[-1])

def build_sorted_ts_index(paths):
    """
    Returns (ts_sorted, order) where:
      ts_sorted = timestamps sorted ascending (np.int64)
      order = indices into 'paths' such that paths[order[k]] has timestamp ts_sorted[k]
    """
    ts = np.array([ts_ns_from_path(p) for p in paths], dtype=np.int64)
    order = np.argsort(ts)
    return ts[order], order

def nearest_idx(sorted_ts, x):
    """
    Nearest neighbor index in a sorted 1D array (np.searchsorted-based).
    """
    i = int(np.searchsorted(sorted_ts, x))
    if i <= 0:
        return 0
    if i >= len(sorted_ts):
        return len(sorted_ts) - 1
    before = sorted_ts[i - 1]
    after = sorted_ts[i]
    return (i - 1) if (x - before) <= (after - x) else i


# -------------------------------------------------------------------------
# Load intrinsics + cam2lidar from calibration_*.json
# -------------------------------------------------------------------------

def load_calibration_bundle():
    """
    Reads:
      - calibration_ar0820.json
      - calibration_cam_stereo.json
      - calibration_gated.json

    Returns:
      intrinsics: {
          'ar0820_left': {'K': 3x3, 'd': [...]},
          'ar0820_right': {...},
          'rgb_left': {...},
          'rgb_right': {...},
          'gated_left': {...},
          'gated_right': {...},
          '_raw': {...}
      }

      extrinsics: {
          'ar0820_left': {
              'T_cam2lidar': 4x4 list,
              'T_vec': [tx,ty,tz],
              'R_quat': [qx,qy,qz,qw],
          },
          'ar0820_right': {...},
          'rgb_left': {...},
          'rgb_right': {...},
          'gated_left': {...},
          'gated_right': {...},
      }
    """
    intrinsics = {}
    extrinsics = {}
    raw = {}

    for cam_group_key, fpath in CALIBRATION_FILES.items():
        if not os.path.isfile(fpath):
            # file not present: fill with None
            intrinsics[f"{cam_group_key}_left"] = None
            intrinsics[f"{cam_group_key}_right"] = None
            extrinsics[f"{cam_group_key}_left"] = None
            extrinsics[f"{cam_group_key}_right"] = None
            raw[cam_group_key] = None
            continue

        data = _load_json(fpath)
        raw[cam_group_key] = data

        # ----- intrinsics -----
        K_l = data.get("K_l")
        K_r = data.get("K_r")
        d_l = data.get("d_l")
        d_r = data.get("d_r")

        K_l_mat = (
            np.array(K_l, dtype=np.float64).reshape(3, 3).tolist()
            if K_l is not None
            else None
        )
        K_r_mat = (
            np.array(K_r, dtype=np.float64).reshape(3, 3).tolist()
            if K_r is not None
            else None
        )

        intrinsics[f"{cam_group_key}_left"] = {"K": K_l_mat, "d": d_l}
        intrinsics[f"{cam_group_key}_right"] = {"K": K_r_mat, "d": d_r}

        # ----- extrinsics cam->lidar -----
        # file: lidar -> left_cam  -> invert -> left_cam -> lidar
        q_lidar2l = data.get("R_l2lidar")
        t_lidar2l = data.get("T_l2lidar")
        R_l2lidar, T_l2lidar = (q_lidar2l, t_lidar2l)

        mat33 = R.from_quat(np.array(R_l2lidar)).as_matrix()
        T = np.eye(4, 4)
        T[:3, :3] = mat33
        T[:3, -1] = T_l2lidar
        T_l_mat = np.linalg.inv(T)

        # file: lidar -> right_cam -> invert -> right_cam -> lidar
        R_r2lidar, T_r2lidar = None, None  # invert_qt_xyzw(q_lidar2r, t_lidar2r)

        # file: right_cam -> left_cam -> invert -> left_cam -> right_cam
        q_r2l = data.get("R_l2r")
        t_r2l = data.get("T_l2r")
        R_l2r, T_l2r = (q_r2l, t_r2l)
        mat33 = R.from_quat(np.array(R_l2r)).as_matrix()
        T = np.eye(4, 4)
        T[:3, :3] = mat33
        T[:3, -1] = T_l2r
        T_l2r_mat = T

        T_r_mat_direct = None

        # ----- LEFT camera extrinsics -----
        extrinsics[f"{cam_group_key}_left"] = (
            {
                "T_cam2lidar": T_l_mat.tolist() if T_l_mat is not None else None,
                "T_vec": T_l2lidar,
                "R_quat": R_l2lidar,
            }
            if (T_l_mat is not None)
            else None
        )

        # ----- RIGHT camera extrinsics -----
        # Prefer explicit T_r2lidar from JSON if present;
        # otherwise compute via chaining: T_r2lidar = T_l2lidar @ inv(T_l2r).
        if T_r_mat_direct is not None:
            # we have explicit right->lidar
            T_r_mat = T_r_mat_direct
            t_vec_r = T_r2lidar
            r_quat_r = R_r2lidar
        elif T_l_mat is not None and T_l2r_mat is not None:
            # derive right->lidar from left->lidar and left->right
            T_r_mat = T_l_mat @ np.linalg.inv(T_l2r_mat)
            t_vec_r = T_r_mat[:3, 3].tolist()
            r_quat_r = rot_to_quat(T_r_mat[:3, :3])
        else:
            T_r_mat = None
            t_vec_r = None
            r_quat_r = None

        extrinsics[f"{cam_group_key}_right"] = (
            {
                "T_cam2lidar": T_r_mat.tolist() if T_r_mat is not None else None,
                "T_vec": t_vec_r,
                "R_quat": r_quat_r,
            }
            if T_r_mat is not None
            else None
        )

    intrinsics["_raw"] = raw
    return intrinsics, extrinsics


# -------------------------------------------------------------------------
# Main loader (timestamp logic kept)
# -------------------------------------------------------------------------

def load_file_dict_multi_all(
    split_file=(
        "splits/gated_stereo/test_day_gatedstereo.txt"
    ),
    opt=None,
    data_path="",
):
    """
    Refactor of load_file_list_multi_all:

    - Keeps original timestamp-based matching logic (gated type9 as reference).
    - Uses ar0820 (RCCB), gated, and rgb (cam_stereo).
    - Extrinsics cam->lidar from calibration_*.json (T_l2lidar / R_l2lidar etc).
    - Returns a dict:
        {
          'root': ...,
          'split_file': ...,
          'calibration': {'intrinsics': ..., 'extrinsics': ...},
          'samples': [ sample_dict, ... ]
        }
    """

    # --------- dataset root ---------
    if opt is not None:
        root = opt.data_path
    else:
        root = data_path

    last_folder_name = os.path.basename(root.rstrip("/"))

    # --------- load split file ---------
    set_training = set()
    with open(split_file, "r") as f:
        lines = [line.rstrip() for line in f.readlines()]
    for l in lines:
        day, ind = l.split(",")
        set_training.add((day, ind))

    print("len_set = {}".format(len(set_training)))

    all_folders = glob(os.path.join(root, "*/"))
    allowed_recordings = set(day for (day, ind) in set_training)
    folders = [
        f for f in all_folders
        if os.path.basename(f.rstrip("/")) in allowed_recordings
    ]

    # --------- load calibration (intrinsics + extrinsics from calibration_*.json) ---------
    intrinsics, extrinsics = load_calibration_bundle()

    samples = []

    # --------- main loop over recordings ---------
    for folder in tqdm(folders):
        # ar0820 RCCB (rectified)
        left_p_ar0820 = os.path.join(folder, "ar0820/left/image_raw/*.png")
        right_p_ar0820 = os.path.join(folder, "ar0820/right/image_raw/*.png")
        list_left_ar0820 = sorted(glob(left_p_ar0820))
        list_right_ar0820 = sorted(glob(right_p_ar0820))

        # Gated (rectified exposures; 5 per side)
        left_p_8 = os.path.join(folder, "framegrabber/left/bwv/type8/image_raw/*.png")
        left_p_9 = os.path.join(folder, "framegrabber/left/bwv/type9/image_raw/*.png")
        left_p_10 = os.path.join(folder, "framegrabber/left/bwv/type10/image_raw/*.png")
        left_passive_1_p = os.path.join(folder, "framegrabber/left/bwv/type6/image_raw/*.png")
        left_passive_2_p = os.path.join(folder, "framegrabber/left/bwv/type7/image_raw/*.png")

        right_p_8 = os.path.join(folder, "framegrabber/right/bwv/type8/image_raw/*.png")
        right_p_9 = os.path.join(folder, "framegrabber/right/bwv/type9/image_raw/*.png")
        right_p_10 = os.path.join(folder, "framegrabber/right/bwv/type10/image_raw/*.png")
        right_passive_1_p = os.path.join(folder, "framegrabber/right/bwv/type6/image_raw/*.png")
        right_passive_2_p = os.path.join(folder, "framegrabber/right/bwv/type7/image_raw/*.png")

        list_left_p_8 = sorted(glob(left_p_8))
        list_left_p_9 = sorted(glob(left_p_9))
        list_left_p_10 = sorted(glob(left_p_10))
        list_left_passive_1_p = sorted(glob(left_passive_1_p))
        list_left_passive_2_p = sorted(glob(left_passive_2_p))

        list_right_p_8 = sorted(glob(right_p_8))
        list_right_p_9 = sorted(glob(right_p_9))
        list_right_p_10 = sorted(glob(right_p_10))
        list_right_passive_1_p = sorted(glob(right_passive_1_p))
        list_right_passive_2_p = sorted(glob(right_passive_2_p))

        # RGB (cam_stereo, rectified)
        left_p_rgb = os.path.join(folder, "cam_stereo/left/image_raw/*.png")
        right_p_rgb = os.path.join(folder, "cam_stereo/right/image_raw/*.png")
        list_left_rgb = sorted(glob(left_p_rgb))
        list_right_rgb = sorted(glob(right_p_rgb))

        # Lidar projections
        left_ar0820_lidar_p = os.path.join(folder, "ar0820/left/lidar_vls128_projected/*.npz")
        left_gated_lidar_p = os.path.join(folder, "framegrabber/left/lidar_vls128_projected/*.npz")
        left_rgb_lidar_p = os.path.join(folder, "cam_stereo/left/lidar_vls128_projected/*.npz")
        raw_lidar_p = os.path.join(folder, "lidar_vls128/velodyne_points_0/*.bin")

        list_left_ar0820_lidar = sorted(glob(left_ar0820_lidar_p))
        list_left_gated_lidar = sorted(glob(left_gated_lidar_p))
        list_left_rgb_lidar = sorted(glob(left_rgb_lidar_p))
        list_raw_lidar = sorted(glob(raw_lidar_p))


        # Optional: rgb may or may not be present, we handle that separately
        has_rgb = (len(list_left_rgb) > 0) and (len(list_right_rgb) > 0)

        # --- compute image shapes ONCE per recording (folder) ---
        ar0820_left_shape = get_image_shape_header(list_left_ar0820[0]) if len(list_left_ar0820) > 0 else None
        gated_left_shape = get_image_shape_header(list_left_p_9[0]) if len(list_left_p_9) > 0 else None
        rgb_left_shape_folder = get_image_shape_header(list_left_rgb[0]) if has_rgb else None

        # --- build sorted timestamp indices for fast nearest matching ---
        ar0820_ts_sorted, ar0820_order = build_sorted_ts_index(list_left_ar0820)
        ar0820_ts_sorted_right, ar0820_order_right = build_sorted_ts_index(list_right_ar0820)
        lidar_ar0820_ts_sorted, lidar_ar0820_order = (
            build_sorted_ts_index(list_left_ar0820_lidar)
            if len(list_left_ar0820_lidar) > 0
            else (None, None)
        )
        lidar_gated_ts_sorted, lidar_gated_order = (
            build_sorted_ts_index(list_left_gated_lidar)
            if len(list_left_gated_lidar) > 0
            else (None, None)
        )

        lidar_raw_ts_sorted, lidar_raw_order = (
            build_sorted_ts_index(list_raw_lidar)
            if len(list_raw_lidar) > 0
            else (None, None)
        )

        if has_rgb:
            rgb_ts_sorted, rgb_order = build_sorted_ts_index(list_left_rgb)
            rgb_ts_sorted_right, rgb_order_right = build_sorted_ts_index(list_right_rgb)
            lidar_rgb_ts_sorted, lidar_rgb_order = (
                build_sorted_ts_index(list_left_rgb_lidar)
                if len(list_left_rgb_lidar) > 0
                else (None, None)
            )
        else:
            rgb_ts_sorted, rgb_order = None, None
            lidar_rgb_ts_sorted, lidar_rgb_order = None, None

        # ---- iterate over gated images and match nearest ar0820 + lidar (+ rgb) ----
        for i in range(len(list_left_p_9)):
            # recording/day + index
            day = list_left_p_9[i].split("/" + last_folder_name + "/")[1].split("/")[0]
            ind = list_left_p_9[i].split("/")[-1].split("_")[0]

            if (day, ind) in set_training:
                # gated reference timestamp (ns int)
                t_gated_ns = ts_ns_from_path(list_left_p_9[i])

                # nearest ar0820 frame (fast)
                ar0820_i_sorted = nearest_idx(ar0820_ts_sorted, t_gated_ns)
                ar0820_i = int(ar0820_order[ar0820_i_sorted])

                ar0820_i_sorted_right = nearest_idx(ar0820_ts_sorted_right, t_gated_ns)
                ar0820_i_right = int(ar0820_order_right[ar0820_i_sorted_right])

                # nearest lidar frames (fast)
                if lidar_ar0820_ts_sorted is not None:
                    pc_i_ar0820_sorted = nearest_idx(lidar_ar0820_ts_sorted, t_gated_ns)
                    pc_i_ar0820 = int(lidar_ar0820_order[pc_i_ar0820_sorted])
                else:
                    pc_i_ar0820 = 0

                if lidar_gated_ts_sorted is not None:
                    pc_i_gated_sorted = nearest_idx(lidar_gated_ts_sorted, t_gated_ns)
                    pc_i_gated = int(lidar_gated_order[pc_i_gated_sorted])
                else:
                    pc_i_gated = 0

                if lidar_raw_ts_sorted is not None:
                    pc_i_raw_sorted = nearest_idx(lidar_raw_ts_sorted, t_gated_ns)
                    pc_i_raw = int(lidar_raw_order[pc_i_raw_sorted])
                else:
                    pc_i_raw = 0

                # optional nearest rgb frame + lidar (fast)
                if has_rgb:
                    rgb_i_sorted = nearest_idx(rgb_ts_sorted, t_gated_ns)
                    rgb_i = int(rgb_order[rgb_i_sorted])

                    rgb_i_sorted_right = nearest_idx(rgb_ts_sorted_right, t_gated_ns)
                    rgb_i_right = int(rgb_order_right[rgb_i_sorted_right])


                    if lidar_rgb_ts_sorted is not None:
                        pc_i_rgb_sorted = nearest_idx(lidar_rgb_ts_sorted, t_gated_ns)
                        pc_i_rgb = int(lidar_rgb_order[pc_i_rgb_sorted])
                        lidar_left_rgb = list_left_rgb_lidar[pc_i_rgb]
                    else:
                        lidar_left_rgb = None

                    image_left_rgb = list_left_rgb[rgb_i]
                    image_right_rgb = list_right_rgb[rgb_i_right]
                    rgb_left_shape = rgb_left_shape_folder
                else:
                    image_left_rgb = None
                    image_right_rgb = None
                    lidar_left_rgb = None
                    rgb_left_shape = None

                # paths (ar0820 + gated, like before)
                image_left_ar0820 = list_left_ar0820[ar0820_i]
                image_right_ar0820 = list_right_ar0820[ar0820_i_right]

                image_left_8 = list_left_p_8[i]
                image_left_9 = list_left_p_9[i]
                image_left_10 = list_left_p_10[i]
                image_left_passive_1 = list_left_passive_1_p[i]
                image_left_passive_2 = list_left_passive_2_p[i]

                image_right_8 = list_right_p_8[i]
                image_right_9 = list_right_p_9[i]
                image_right_10 = list_right_p_10[i]
                image_right_passive_1 = list_right_passive_1_p[i]
                image_right_passive_2 = list_right_passive_2_p[i]

                lidar_left_ar0820 = list_left_ar0820_lidar[pc_i_ar0820] if len(list_left_ar0820_lidar) > 0 else None
                lidar_left_gated = list_left_gated_lidar[pc_i_gated] if len(list_left_gated_lidar) > 0 else None
                raw_lidar = list_raw_lidar[pc_i_raw] if len(list_raw_lidar) > 0 else None

                name = f"{day}_{ind}"

                # intrinsics/extrinsics
                ar0820_left_intr = intrinsics.get("ar0820_left")
                ar0820_right_intr = intrinsics.get("ar0820_right")
                ar0820_left_extr = extrinsics.get("ar0820_left")
                ar0820_right_extr = extrinsics.get("ar0820_right")

                gated_left_intr = intrinsics.get("gated_left")
                gated_right_intr = intrinsics.get("gated_right")
                gated_left_extr = extrinsics.get("gated_left")
                gated_right_extr = extrinsics.get("gated_right")

                rgb_left_intr = intrinsics.get("rgb_left")
                rgb_right_intr = intrinsics.get("rgb_right")
                rgb_left_extr = extrinsics.get("rgb_left")
                rgb_right_extr = extrinsics.get("rgb_right")

                # ---- build nested sample dict ----
                cam_dict = {
                    # RCCB camera = ar0820
                    "ar0820": {
                        "left": {
                            "image": image_left_ar0820,
                            "intrinsics": ar0820_left_intr,
                            "cam2lidar": (
                                ar0820_left_extr["T_cam2lidar"]
                                if ar0820_left_extr is not None
                                else None
                            ),
                            "image_shape": ar0820_left_shape,
                        },
                        "right": {
                            "image": image_right_ar0820,
                            "intrinsics": ar0820_right_intr,
                            "cam2lidar": (
                                ar0820_right_extr["T_cam2lidar"]
                                if ar0820_right_extr is not None
                                else None
                            ),
                            "image_shape": ar0820_left_shape,
                        },
                    },
                    # gated camera, 5 exposures per side
                    "gated": {
                        "left": {
                            "type8": {
                                "image": image_left_8,
                                "intrinsics": gated_left_intr,
                                "cam2lidar": (
                                    gated_left_extr["T_cam2lidar"]
                                    if gated_left_extr is not None
                                    else None
                                ),
                                "image_shape": gated_left_shape,
                            },
                            "type9": {
                                "image": image_left_9,
                                "intrinsics": gated_left_intr,
                                "cam2lidar": (
                                    gated_left_extr["T_cam2lidar"]
                                    if gated_left_extr is not None
                                    else None
                                ),
                                "image_shape": gated_left_shape,
                            },
                            "type10": {
                                "image": image_left_10,
                                "intrinsics": gated_left_intr,
                                "cam2lidar": (
                                    gated_left_extr["T_cam2lidar"]
                                    if gated_left_extr is not None
                                    else None
                                ),
                                "image_shape": gated_left_shape,
                            },
                            "passive1": {
                                "image": image_left_passive_1,
                                "intrinsics": gated_left_intr,
                                "cam2lidar": (
                                    gated_left_extr["T_cam2lidar"]
                                    if gated_left_extr is not None
                                    else None
                                ),
                                "image_shape": gated_left_shape,
                            },
                            "passive2": {
                                "image": image_left_passive_2,
                                "intrinsics": gated_left_intr,
                                "cam2lidar": (
                                    gated_left_extr["T_cam2lidar"]
                                    if gated_left_extr is not None
                                    else None
                                ),
                                "image_shape": gated_left_shape,
                            },
                        },
                        "right": {
                            "type8": {
                                "image": image_right_8,
                                "intrinsics": gated_right_intr,
                                "cam2lidar": (
                                    gated_right_extr["T_cam2lidar"]
                                    if gated_right_extr is not None
                                    else None
                                ),
                                "image_shape": gated_left_shape,
                            },
                            "type9": {
                                "image": image_right_9,
                                "intrinsics": gated_right_intr,
                                "cam2lidar": (
                                    gated_right_extr["T_cam2lidar"]
                                    if gated_right_extr is not None
                                    else None
                                ),
                                "image_shape": gated_left_shape,
                            },
                            "type10": {
                                "image": image_right_10,
                                "intrinsics": gated_right_intr,
                                "cam2lidar": (
                                    gated_right_extr["T_cam2lidar"]
                                    if gated_right_extr is not None
                                    else None
                                ),
                                "image_shape": gated_left_shape,
                            },
                            "passive1": {
                                "image": image_right_passive_1,
                                "intrinsics": gated_right_intr,
                                "cam2lidar": (
                                    gated_right_extr["T_cam2lidar"]
                                    if gated_right_extr is not None
                                    else None
                                ),
                                "image_shape": gated_left_shape,
                            },
                            "passive2": {
                                "image": image_right_passive_2,
                                "intrinsics": gated_right_intr,
                                "cam2lidar": (
                                    gated_right_extr["T_cam2lidar"]
                                    if gated_right_extr is not None
                                    else None
                                ),
                                "image_shape": gated_left_shape,
                            },
                        },
                    },
                }

                # Add RGB block only if we actually have RGB images for this folder
                if has_rgb:
                    cam_dict["rgb"] = {
                        "left": {
                            "image": image_left_rgb,
                            "intrinsics": rgb_left_intr,
                            "cam2lidar": (
                                rgb_left_extr["T_cam2lidar"]
                                if rgb_left_extr is not None
                                else None
                            ),
                            "image_shape": rgb_left_shape,
                        },
                        "right": {
                            "image": image_right_rgb,
                            "intrinsics": rgb_right_intr,
                            "cam2lidar": (
                                rgb_right_extr["T_cam2lidar"]
                                if rgb_right_extr is not None
                                else None
                            ),
                            "image_shape": rgb_left_shape,
                        },
                    }

                lidar_dict = {
                    "left_ar0820": lidar_left_ar0820,
                    "left_gated": lidar_left_gated,
                    "raw_lidar": raw_lidar,
                }
                if has_rgb:
                    lidar_dict["left_rgb"] = lidar_left_rgb

                sample_dict = {
                    "recording": day,
                    "index": ind,
                    "name": name,
                    "folder": folder.rstrip("/"),
                    "camera": cam_dict,
                    "lidar": lidar_dict,
                }

                samples.append(sample_dict)



    dataset_dict = {
        "root": root,
        "split_file": split_file,
        "calibration": {
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,  # cam->lidar from calibration_*.json
        },
        "samples": samples,
    }

    print("len_samples = {}".format(len(samples)))
    return dataset_dict


if __name__ == "__main__":
    # default paths (change if you want to override)
    parser = argparse.ArgumentParser(description="Build dataset_dict and save as a pickle file.")
    parser.add_argument(
        "--split_file",
        type=str,
        default="splits/test_day_gatedstereo.txt",
        help="Path to split file (txt).",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Root path to dataset.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="file_lists",
        help="Output directory for the generated .pkl file.",
    )
    # optional: override calibration jsons without editing code
    parser.add_argument("--calib_ar0820", type=str, default=None, help="Override ar0820 calibration JSON.")
    parser.add_argument("--calib_rgb", type=str, default=None, help="Override rgb calibration JSON.")
    parser.add_argument("--calib_gated", type=str, default=None, help="Override gated calibration JSON.")

    args = parser.parse_args()

    # Optionally override calibration files from CLI
    if args.calib_ar0820 is not None:
        CALIBRATION_FILES["ar0820"] = args.calib_ar0820
    if args.calib_rgb is not None:
        CALIBRATION_FILES["rgb"] = args.calib_rgb
    if args.calib_gated is not None:
        CALIBRATION_FILES["gated"] = args.calib_gated

    dataset_dict = load_file_dict_multi_all(
        split_file=args.split_file,
        data_path=args.data_path,
        opt=None,  # keep None unless you have an opt object with .data_path
    )

    os.makedirs(args.out_dir, exist_ok=True)

    split_base = os.path.basename(args.split_file)  # e.g. "test_day_gatedstereo.txt"
    split_stem, _ = os.path.splitext(split_base)  # e.g. "test_day_gatedstereo"
    out_path = os.path.join(args.out_dir, split_stem + ".pkl")

    with open(out_path, "wb") as f:
        pickle.dump(dataset_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved dataset_dict with {len(dataset_dict['samples'])} samples to {out_path}")
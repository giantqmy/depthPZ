import os
import json
import copy
from abc import ABCMeta
from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple, Any
from Raw2RGB.main import generate_rgb_image
import tifffile as tiff
from collections import defaultdict
import re

import numpy as np
import cv2
import torch


from dataset.data_samples import (
    PointData, InstanceData, BasePoints, BaseInstance3DBoxes
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *a, **kw):
        return x

class BaseTransform(metaclass=ABCMeta):
    """Base class for all transformations."""

    def __call__(self,
                 results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:

        return self.transform(results)


class AddDatasetKey(BaseTransform):
    """
    Adds results['dataset'].

    Priority:
      1) If results already contains 'dataset', keep it (optionally alias).
      2) Else infer from results['recording'] prefix:
         - "_"    -> "ms2"
         - "2022" -> "gatedstereo"
    """

    def __init__(self, store_key: str = "dataset"):
        super().__init__()
        self.store_key = store_key

        # optional aliases if your code expects specific names
        self.alias = {
            "kitti_completion": "kitti_dc",
            "kitti": "kitti_dc",
            "ms2": "ms2",
            "gatedstereo": "gatedstereo",
        }

    def transform(self, results: dict):
        # 1) Respect pre-set dataset (from annotation file)
        ds = results.get(self.store_key, None)
        if isinstance(ds, str) and ds:
            results[self.store_key] = self.alias.get(ds, ds)
            return results

        # 2) Otherwise infer from recording
        rec = results.get("recording", None)
        if isinstance(rec, str):
            if rec.startswith("_"):
                results[self.store_key] = "ms2"
            elif rec.startswith("2022"):
                results[self.store_key] = "gatedstereo"
            else:
                # as a last resort, leave unset or set to unknown
                # results[self.store_key] = "unknown"
                pass

        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(store_key='{self.store_key}')"

class GatedStereoLoadPointsFromBin(BaseTransform):
    def __init__(self):
        pass
    def transform(self, results):
        points = np.fromfile(results['lidar']['raw_lidar'], dtype=np.float32).reshape(-1, 5)
        results['points'] = points
        return results


class GatedStereoLoadMultiViewImageFromFiles(BaseTransform):
    """Load multi-view images for the gated-stereo dataset.

    Expects `results` to already contain a sample dict like:

        results["camera"] = {
            "ar0820": {
                "left": {
                    "image": <path>,
                    "intrinsics": {"K": 3x3, "d": [...]},
                    "cam2lidar": 4x4 list,
                    "image_shape": (H,W,C),
                },
                "right": {...},
            },
            "gated": {
                "left": {
                    "type8":  {"image": <path>, "intrinsics": {...}, "cam2lidar": 4x4, ...},
                    "type9":  {...},
                    "type10": {...},
                    "passive1": {...},
                    "passive2": {...},
                },
                "right": {...},
            },
            "rgb": {
                "left": {
                    "image": <path>,
                    "intrinsics": {"K": 3x3, "d": [...]},
                    "cam2lidar": 4x4 list,
                    "image_shape": (H,W,C),
                },
                "right": {...},
            },
        }

        results["lidar"] = {
            "left_ar0820": <npz path>,
            "left_gated": <npz path>,
            "left_rgb": <npz path> or None,
            "raw_lidar": <.bin path> or None,
        }

    Cameras are referenced in the config with names:
      - gated_left, gated_right,
      - rccb_left, rccb_right,
      - rgb_left, rgb_right.

    For gated_* cameras, all 5 exposures
        [type8, type9, type10, passive1, passive2]
    are loaded and concatenated along channel dimension → 5-channel image
    (assuming each exposure is single-channel).
    """

    _CAMERA_MAPPING = {
        "gated_left": ("gated", "left"),
        "gated_right": ("gated", "right"),
        "rccb_left": ("ar0820", "left"),
        "rccb_right": ("ar0820", "right"),
        "rgb_left": ("rgb", "left"),
        "rgb_right": ("rgb", "right"),
    }

    _GATED_EXPOSURES = ["type8", "type9", "type10", "passive1", "passive2"]

    def __init__(
        self,
        to_float32: bool = False,
        color_type: str = "unchanged",  # 'color', 'grayscale', 'unchanged'
        num_views_to_sample: int = 4,
        sample_probability: Optional[Dict[str, float]] = None,
        set_default_scale: bool = True,
    ) -> None:
        super().__init__()
        self.to_float32 = to_float32
        self.color_type = color_type
        self.set_default_scale = set_default_scale

        # sampling: camera tags from config (subset of the 6 supported)
        sample_probability = sample_probability or {}
        if sample_probability:
            self.camera_tags: List[str] = list(sample_probability.keys())
            self.camera_probs: List[float] = list(sample_probability.values())
        else:
            # default: all supported, uniform
            self.camera_tags = list(self._CAMERA_MAPPING.keys())
            self.camera_probs = []

        self.num_cams_to_sample = int(num_views_to_sample)

    # --- minimal helpers (same style as your original LoadMultiViewImageFromFiles) ---

    def _imread(self, path: str, image_shape) -> np.ndarray:
        """Read image with OpenCV. On failure, return zeros(image_shape) and log path once."""
        if self.color_type == "color":
            flag = cv2.IMREAD_COLOR
        elif self.color_type == "grayscale":
            flag = cv2.IMREAD_GRAYSCALE
        else:  # 'unchanged'
            flag = cv2.IMREAD_UNCHANGED

        try:
            img = cv2.imread(path, flag)
            if img is None:
                raise ValueError("cv2.imread returned None")

            # Convert BGR->RGB for 'color' to mimic mmcv
            if flag == cv2.IMREAD_COLOR:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

        except Exception:
            # fallback image
            print(path)
            fallback = np.zeros(image_shape, dtype=np.uint8)

            # log once
            failed_file = "failed_pngs.txt"
            try:
                existing = set()
                if os.path.exists(failed_file):
                    with open(failed_file, "r", encoding="utf-8") as f:
                        existing = {line.strip() for line in f if line.strip()}
                if path not in existing:
                    with open(failed_file, "a", encoding="utf-8") as f:
                        f.write(path + "\n")
            except Exception:
                # don't let logging break the dataloader
                pass

            return fallback

    def _impad_to(self, img: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
        """Pad image to (H, W) with zeros."""
        H, W = img.shape[:2]
        outH, outW = int(shape_hw[0]), int(shape_hw[1])
        if H == outH and W == outW:
            return img

        if img.ndim == 2:
            out = np.zeros((outH, outW), dtype=img.dtype)
            out[:H, :W] = img
        else:
            C = img.shape[2]
            out = np.zeros((outH, outW, C), dtype=img.dtype)
            out[:H, :W, :] = img
        return out

    # --- camera selection / sampling ---

    def _available_cameras(self, results: dict) -> List[str]:
        """List camera tags that exist in results['camera'] and have image paths."""
        cams = results.get("camera", {})
        available = []
        for tag in self.camera_tags:
            if tag not in self._CAMERA_MAPPING:
                continue
            group, side = self._CAMERA_MAPPING[tag]
            if group not in cams:
                continue
            side_dict = cams[group].get(side)
            if side_dict is None:
                continue

            # gated_*: require all 5 exposures
            if group == "gated":
                if all(
                    side_dict.get(exp, {}).get("image", None) is not None
                    for exp in self._GATED_EXPOSURES
                ):
                    available.append(tag)
            else:
                # rccb/rgb: single image key 'image'
                if side_dict.get("image", None) is not None:
                    available.append(tag)
        return available

    def _sample_cameras(self, available: List[str]) -> List[str]:
        """Randomly sample cameras from 'available' using configured probs."""
        n_avail = len(available)
        if n_avail == 0:
            raise ValueError("No available cameras in sample (results['camera']).")

        k = min(self.num_cams_to_sample, n_avail)
        if k == n_avail:
            return available

        # build prob vector aligned with 'available', if user provided probabilities
        p = None
        if self.camera_probs:
            name_to_prob = {k: v for k, v in zip(self.camera_tags, self.camera_probs)}
            probs = np.array([name_to_prob.get(tag, 0.0) for tag in available], dtype=np.float64)
            s = probs.sum()
            if s > 0:
                p = probs / s

        idx = np.random.choice(n_avail, size=k, replace=False, p=p)
        return [available[i] for i in idx]

    # --- image loading per camera ---

    def _load_raw_rccb_image(self, path):
        bgr = generate_rgb_image(path, camtype='ar0820')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def _load_raw_rgb_image(self, path):
        """
        Takes a raw data image and converts it to a 8 bit RGB image with for visual inspection.
        Images are only rectified and bitshifted.
        """
        image_raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        image_bayer = cv2.cvtColor(image_raw, cv2.COLOR_BAYER_GB2BGR)
        image_bit = np.right_shift(image_bayer, 4).astype(np.uint8)

        return image_bit

    def _imread_raw_gated(self, path):
        return (np.expand_dims(tiff.imread(path), 2) / 4).astype(np.uint8)

    def _load_gated_image(self, side_dict: dict) -> np.ndarray:
        """Load 5 gated exposures and concatenate along channel dimension."""
        imgs = []
        paths = []
        for exp in self._GATED_EXPOSURES:
            entry = side_dict.get(exp)
            if entry is None or "image" not in entry:
                raise ValueError(f"Missing gated exposure '{exp}' in sample.")
            path = entry["image"]
            paths.append(path)
            if 'tiff' in path:
                imgs.append(self._imread_raw_gated(path))
            else:
                imgs.append(self._imread(path, side_dict[exp]['image_shape']))

        # sanity: HxW equal
        shapes = [im.shape for im in imgs]
        if len({s[:2] for s in shapes}) != 1:
            raise ValueError(f"Gated exposures have different sizes: {shapes}")

        # grayscale -> H x W x 5, color -> H x W x (C*5)
        if imgs[0].ndim == 2:
            stacked = np.stack(imgs, axis=-1)  # H x W x 5
        else:
            stacked = np.concatenate(imgs, axis=-1)  # H x W x (C*5)
        return stacked

    def _load_single_image(self, path: str, image_shape) -> np.ndarray:
        return self._imread(path, image_shape)

    # --- main transform ---

    def transform(self, results: dict) -> Optional[dict]:
        # 1) determine which cameras exist and sample subset
        available = self._available_cameras(results)
        selected_tags = self._sample_cameras(available)
        cams = results["camera"]

        # group-wise containers: 'gated', 'rgb', 'rccb'
        group_names = ["gated", "rgb", "rccb"]
        group_tags = {g: [] for g in group_names}
        group_imgs = {g: [] for g in group_names}
        group_filenames = {g: [] for g in group_names}
        group_cam2img = {g: [] for g in group_names}
        group_cam2lidar = {g: [] for g in group_names}
        group_lidar2cam = {g: [] for g in group_names}
        group_lidar2img = {g: [] for g in group_names}

        # global flattened containers
        flat_tags: List[str] = []
        flat_imgs: List[np.ndarray] = []
        flat_filenames: List[str] = []
        flat_cam2img: List[np.ndarray] = []
        flat_cam2lidar: List[np.ndarray] = []
        flat_lidar2cam: List[np.ndarray] = []
        flat_lidar2img: List[np.ndarray] = []

        for tag in selected_tags:
            group_raw, side = self._CAMERA_MAPPING[tag]  # 'gated', 'ar0820', 'rgb'
            # expose ar0820 as 'rccb' group in results
            group = "rccb" if group_raw == "ar0820" else group_raw

            side_dict = cams[group_raw][side]

            if group_raw == "gated":
                # multi-exposure 5-channel image
                img = self._load_gated_image(side_dict)
                # take intrinsics / extrinsics from, e.g., type9
                calib_entry = side_dict["type9"]
                filename = side_dict["type9"]["image"]
            elif group_raw == 'ar0820':
                if 'tiff' in side_dict["image"]:
                    img = self._load_raw_rccb_image(side_dict["image"])
                else:
                    img = self._load_single_image(side_dict["image"], side_dict['image_shape'])
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                calib_entry = side_dict
                filename = calib_entry["image"]
            elif group_raw == 'rgb':
                if 'tiff' in side_dict["image"]:
                    img = self._load_raw_rgb_image(side_dict["image"])
                else:
                    img = self._load_single_image(side_dict["image"], side_dict['image_shape'])
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                calib_entry = side_dict
                filename = calib_entry["image"]
            else:
                img = self._load_single_image(side_dict["image"], side_dict['image_shape'])
                calib_entry = side_dict
                filename = calib_entry["image"]

            # --- intrinsics -> cam2img (4x4) ---
            intr = calib_entry.get("intrinsics")
            if intr is None or intr.get("K") is None:
                raise ValueError(f"Missing intrinsics for camera {tag}")
            K = np.array(intr["K"], dtype=np.float32).reshape(3, 3)
            cam2img = np.eye(4, dtype=np.float32)
            cam2img[:3, :3] = K

            # --- extrinsics cam->lidar (4x4) ---
            cam2lidar = np.array(calib_entry["cam2lidar"], dtype=np.float32)
            lidar2cam = np.linalg.inv(cam2lidar)
            lidar2img = cam2img @ lidar2cam

            # append to group-level containers
            group_tags[group].append(tag)
            group_imgs[group].append(img)
            group_filenames[group].append(filename)
            group_cam2img[group].append(cam2img)
            group_cam2lidar[group].append(cam2lidar)
            group_lidar2cam[group].append(lidar2cam)
            group_lidar2img[group].append(lidar2img)

            # append to global containers
            flat_tags.append(tag)
            flat_imgs.append(img)
            flat_filenames.append(filename)
            flat_cam2img.append(cam2img)
            flat_cam2lidar.append(cam2lidar)
            flat_lidar2cam.append(lidar2cam)
            flat_lidar2img.append(lidar2img)

        num_views = len(flat_tags)

        # 2) build per-group dicts in results: results['gated'], results['rgb'], results['rccb']
        for g in group_names:
            if len(group_filenames[g]) == 0:
                results[g] = None
                continue

            results[g] = dict(
                view_names=group_tags[g],
                filename=group_filenames[g],
                img=group_imgs[g],  # list of HxWxC arrays, possibly different shapes
                img_shape=[im.shape for im in group_imgs[g]],
                cam2img=np.stack(group_cam2img[g], axis=0),
                cam2lidar=np.stack(group_cam2lidar[g], axis=0),
                lidar2cam=np.stack(group_lidar2cam[g], axis=0),
                lidar2img=np.stack(group_lidar2img[g], axis=0),
            )

        # 3) flatten into "global" keys (no padding / stacking across views!)
        results["view_names"] = flat_tags
        results["filename"] = flat_filenames
        results["img"] = flat_imgs  # list of np.ndarray, varying shapes / channels
        results["img_shape"] = [im.shape for im in flat_imgs]
        results["ori_shape"] = results["img_shape"]
        results["pad_shape"] = results["img_shape"]
        results["num_views"] = num_views

        # 4) camera / lidar transforms (stacked over all selected views)
        cam2img_arr = np.stack(flat_cam2img, axis=0)
        cam2lidar_arr = np.stack(flat_cam2lidar, axis=0)
        lidar2cam_arr = np.stack(flat_lidar2cam, axis=0)
        lidar2img_arr = np.stack(flat_lidar2img, axis=0)

        results["cam2img"] = cam2img_arr
        results["cam2lidar"] = cam2lidar_arr
        results["lidar2cam"] = lidar2cam_arr
        results["lidar2img"] = lidar2img_arr
        results["ori_cam2img"] = copy.deepcopy(cam2img_arr)

        # 5) img_norm_cfg: choose max channel count across all views
        channel_counts = []
        for im in flat_imgs:
            if im.ndim == 3:
                channel_counts.append(im.shape[2])
            else:
                channel_counts.append(1)
        max_c = max(channel_counts) if channel_counts else 1
        results["img_norm_cfg"] = dict(
            mean=np.zeros(max_c, dtype=np.float32),
            std=np.ones(max_c, dtype=np.float32),
            to_rgb=False,
        )

        return results


    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"to_float32={self.to_float32}, "
            f"color_type='{self.color_type}', "
            f"num_cams_to_sample={self.num_cams_to_sample}, "
            f"camera_tags={self.camera_tags})"
        )



class RectifyStereoImages(BaseTransform):
    """Rectify stereo pairs for rccb (ar0820), rgb, and gated cameras.

    Expects:
      - results['view_names']: list of camera tags like
            ['gated_left', 'gated_right', 'rccb_left', ...]
      - results['img']: list of images, one per view
      - results['cam2img'], results['cam2lidar'], results['lidar2cam'], results['lidar2img']
        as (N, 4, 4) arrays.
      - results['camera'] tree from the dataset sample, e.g.
            results['camera']['ar0820']['left']['intrinsics']['K']
            results['camera']['gated']['left']['type9']['intrinsics']['d']

    If *_rect_loaded=True for a camera type, we still:
      - update cam2img, lidar2cam, cam2lidar, lidar2img
      - update results['camera'][...]['intrinsics']['K'] and ['cam2lidar']
    but do NOT overwrite results['img'] for that type.
    """

    # mapping from tag in view_names -> camera tree location
    TAG_TO_CAMERA = {
        "rccb_left": ("ar0820", "left"),
        "rccb_right": ("ar0820", "right"),
        "rgb_left": ("rgb", "left"),
        "rgb_right": ("rgb", "right"),
        "gated_left": ("gated", "left"),
        "gated_right": ("gated", "right"),
    }

    GATED_EXPOSURES = ["type8", "type9", "type10", "passive1", "passive2"]

    def __init__(
        self,
        gated_rect_loaded: bool = False,
        rgb_rect_loaded: bool = False,
        rccb_rect_loaded: bool = False,
        alpha_gated: float = -1.0,
        alpha_rgb: float = -1.0,
        alpha_rccb: float = -1.0,

    ):
        super().__init__()
        self.rect_loaded_flags: Dict[str, bool] = {
            "gated": gated_rect_loaded,
            "rgb": rgb_rect_loaded,
            "rccb": rccb_rect_loaded,
        }
        self.alphas = {
            "gated": alpha_gated,
            "rgb": alpha_rgb,
            "rccb": alpha_rccb,
        }

    # ---------------------------------------------------------
    # main API
    # ---------------------------------------------------------
    def transform(self, results: dict) -> Optional[dict]:
        # if it's not our new multi-view structure, just no-op
        if "view_names" not in results or "cam2img" not in results:
            return results

        tag_to_idx = {tag: i for i, tag in enumerate(results["view_names"])}

        # handle each stereo pair type independently
        self._rectify_pair_type(
            results,
            tag_left="rccb_left",
            tag_right="rccb_right",
            group_name="rccb",
            tag_to_idx=tag_to_idx,
            alpha=self.alphas["rccb"],
        )
        self._rectify_pair_type(
            results,
            tag_left="rgb_left",
            tag_right="rgb_right",
            group_name="rgb",
            tag_to_idx=tag_to_idx,
            alpha=self.alphas["rgb"],
        )
        self._rectify_pair_type(
            results,
            tag_left="gated_left",
            tag_right="gated_right",
            group_name="gated",
            tag_to_idx=tag_to_idx,
            alpha=self.alphas["gated"],
        )

        return results

    # ---------------------------------------------------------
    # per-type logic
    # ---------------------------------------------------------
    def _rectify_pair_type(
        self,
        results: dict,
        tag_left: str,
        tag_right: str,
        group_name: str,
        tag_to_idx: Dict[str, int],
        alpha,
    ) -> None:
        """Rectify one stereo pair type if both views are present."""
        if tag_left not in tag_to_idx or tag_right not in tag_to_idx:
            # this stereo pair is not fully available in this sample
            return

        left_idx = tag_to_idx[tag_left]
        right_idx = tag_to_idx[tag_right]

        img_left = results["img"][left_idx]
        img_right = results["img"][right_idx]

        # K from cam2img (top-left 3x3)
        K1 = results["cam2img"][left_idx][:3, :3]
        K2 = results["cam2img"][right_idx][:3, :3]

        # get distortion from results['camera'] if available
        D1, D2 = self._get_distortion(results, tag_left, tag_right)

        # R, T from cam2lidar
        R_lr, T_lr, _ = self.left_to_right_from_cam2lidar(
            results["cam2lidar"][left_idx],
            results["cam2lidar"][right_idx],
        )

        # run rectification to get rectified camera geometry
        rect = self.rectify_pair(
            img_left,
            img_right,
            K1,
            D1,
            K2,
            D2,
            R_lr,
            T_lr,
            alpha=alpha,
        )

        left_rect = rect["left_rect"]
        right_rect = rect["right_rect"]
        K1_new = rect["K1_new"]
        K2_new = rect["K2_new"]
        R1 = rect["R1"]
        R2 = rect["R2"]
        baseline = rect["T_rect"][0:1]

        # should we actually write new images back?
        rect_images = not self.rect_loaded_flags[group_name]

        if rect_images:
            results["img"][left_idx] = left_rect
            results["img"][right_idx] = right_rect
        # else: images assumed already rectified on disk, we only use the
        # rectification to update intrinsics/extrinsics.

        # --- Update lidar<->cam using rectification rotations ---
        R1_4x4 = np.eye(4, dtype=np.float64)
        R1_4x4[:3, :3] = R1
        lidar2cam_left = results["lidar2cam"][left_idx]

        R2_4x4 = np.eye(4, dtype=np.float64)
        R2_4x4[:3, :3] = R2
        lidar2cam_right = results["lidar2cam"][right_idx]

        results["lidar2cam"][left_idx] = R1_4x4 @ lidar2cam_left
        results["lidar2cam"][right_idx] = R2_4x4 @ lidar2cam_right

        # recompute cam2lidar batch-wise
        results["cam2lidar"] = np.linalg.inv(results["lidar2cam"])

        #from scipy.spatial.transform import Rotation as Rot
        #Rot.from_matrix((R1_4x4.T @ lidar2cam_left)[:3, :3]).as_quat()


        # --- Update intrinsics cam2img ---
        results["cam2img"][left_idx][:3, :3] = K1_new
        results["cam2img"][right_idx][:3, :3] = K2_new

        # recompute lidar2img
        results["lidar2img"] = results["cam2img"] @ results["lidar2cam"]

        # store baseline per type (optional)
        results[f"{group_name}_baseline"] = baseline
        results[f"{group_name}_left_idx"] = left_idx
        results[f"{group_name}_right_idx"] = right_idx

        # update the nested 'camera' tree for consistency
        self._update_camera_tree(results, tag_left, tag_right, K1_new, K2_new)

    # ---------------------------------------------------------
    # helpers for intrinsics / distortions / camera tree
    # ---------------------------------------------------------
    def _get_distortion(self, results: dict, tag_left: str, tag_right: str):
        """Fetch distortion vectors D1, D2 from results['camera'] if present."""
        camera_tree = results.get("camera", {})
        D1, D2 = None, None

        # left
        if tag_left in self.TAG_TO_CAMERA:
            group_raw_l, side_l = self.TAG_TO_CAMERA[tag_left]
            if group_raw_l == "gated":
                # choose type9 as canonical
                intr_l = (
                    camera_tree.get("gated", {})
                    .get(side_l, {})
                    .get("type9", {})
                    .get("intrinsics", None)
                )
            else:
                intr_l = (
                    camera_tree.get(group_raw_l, {})
                    .get(side_l, {})
                    .get("intrinsics", None)
                )
            if intr_l is not None:
                D1 = intr_l.get("d", None)

        # right
        if tag_right in self.TAG_TO_CAMERA:
            group_raw_r, side_r = self.TAG_TO_CAMERA[tag_right]
            if group_raw_r == "gated":
                intr_r = (
                    camera_tree.get("gated", {})
                    .get(side_r, {})
                    .get("type9", {})
                    .get("intrinsics", None)
                )
            else:
                intr_r = (
                    camera_tree.get(group_raw_r, {})
                    .get(side_r, {})
                    .get("intrinsics", None)
                )
            if intr_r is not None:
                D2 = intr_r.get("d", None)

        return D1, D2

    def _update_camera_tree(
        self,
        results: dict,
        tag_left: str,
        tag_right: str,
        K1_new: np.ndarray,
        K2_new: np.ndarray,
    ):
        """Write back updated intrinsics & cam2lidar into results['camera'] tree."""
        camera_tree = results.get("camera", None)
        if camera_tree is None:
            return

        cam2lidar_arr = results["cam2lidar"]

        # LEFT
        if tag_left in self.TAG_TO_CAMERA:
            group_raw_l, side_l = self.TAG_TO_CAMERA[tag_left]
            idx_l = results["view_names"].index(tag_left)
            cam2lidar_new_l = cam2lidar_arr[idx_l].tolist()
            if group_raw_l == "gated":
                # update all exposures for this side
                side_node = camera_tree.get("gated", {}).get(side_l, {})
                for exp in self.GATED_EXPOSURES:
                    if exp in side_node:
                        side_node[exp]["cam2lidar"] = cam2lidar_new_l
                        intr = side_node[exp].get("intrinsics", None)
                        if intr is not None:
                            intr["K"] = K1_new.tolist()
            else:
                node = camera_tree.get(group_raw_l, {}).get(side_l, None)
                if node is not None:
                    node["cam2lidar"] = cam2lidar_new_l
                    intr = node.get("intrinsics", None)
                    if intr is not None:
                        intr["K"] = K1_new.tolist()

        # RIGHT
        if tag_right in self.TAG_TO_CAMERA:
            group_raw_r, side_r = self.TAG_TO_CAMERA[tag_right]
            idx_r = results["view_names"].index(tag_right)
            cam2lidar_new_r = cam2lidar_arr[idx_r].tolist()
            if group_raw_r == "gated":
                side_node = camera_tree.get("gated", {}).get(side_r, {})
                for exp in self.GATED_EXPOSURES:
                    if exp in side_node:
                        side_node[exp]["cam2lidar"] = cam2lidar_new_r
                        intr = side_node[exp].get("intrinsics", None)
                        if intr is not None:
                            intr["K"] = K2_new.tolist()
            else:
                node = camera_tree.get(group_raw_r, {}).get(side_r, None)
                if node is not None:
                    node["cam2lidar"] = cam2lidar_new_r
                    intr = node.get("intrinsics", None)
                    if intr is not None:
                        intr["K"] = K2_new.tolist()

    # ---------------------------------------------------------
    # geometry helpers (mostly your originals)
    # ---------------------------------------------------------
    def rectify_pair(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray,
        K1: np.ndarray,
        D1,
        K2: np.ndarray,
        D2,
        R: np.ndarray,
        T: np.ndarray,
        alpha: float = 1.0,
        new_size: Optional[tuple] = None,
        flags: int = cv2.CALIB_ZERO_DISPARITY,
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_CONSTANT,
    ):
        if img_left.shape[:2] != img_right.shape[:2]:
            raise ValueError("Left/right images must share the same HxW.")
        h, w = img_left.shape[:2]
        size = (w, h) if new_size is None else new_size

        K1 = np.asarray(K1, dtype=np.float64)
        K2 = np.asarray(K2, dtype=np.float64)
        R = np.asarray(R, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64).reshape(3, 1)

        def _dist_or_zeros(D):
            if D is None:
                return np.zeros((5, 1), dtype=np.float64)
            D = np.asarray(D, dtype=np.float64).ravel()
            return D.reshape(1, -1)

        D1 = _dist_or_zeros(D1)
        D2 = _dist_or_zeros(D2)

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1,
            D1,
            K2,
            D2,
            (w, h),
            R,
            T,
            flags=flags,
            alpha=alpha,
            newImageSize=size,
        )

        # Rectification maps
        map1_l, map2_l = cv2.initUndistortRectifyMap(
            K1, D1, R1, P1, size, cv2.CV_32FC1
        )
        map1_r, map2_r = cv2.initUndistortRectifyMap(
            K2, D2, R2, P2, size, cv2.CV_32FC1
        )

        # Apply to images (works with multi-channel images too)
        left_rect = cv2.remap(
            img_left,
            map1_l,
            map2_l,
            interpolation=interpolation,
            borderMode=border_mode,
        )
        right_rect = cv2.remap(
            img_right,
            map1_r,
            map2_r,
            interpolation=interpolation,
            borderMode=border_mode,
        )

        # New intrinsics from P1/P2
        K1_new = P1[:, :3]
        K2_new = P2[:, :3]

        # Baseline in rectified coords
        Tx = -P2[0, 3] / P2[0, 0]
        T_rect = np.array([Tx, 0.0, 0.0], dtype=np.float64)
        R_rel_rect = np.eye(3, dtype=np.float64)

        return {
            "left_rect": left_rect,
            "right_rect": right_rect,
            "K1_new": K1_new,
            "K2_new": K2_new,
            "R1": R1,
            "R2": R2,
            "P1": P1,
            "P2": P2,
            "Q": Q,
            "roi1": roi1,
            "roi2": roi2,
            "T_rect": T_rect,
            "R_rel_rect": R_rel_rect,
            # maps in case you want to use them for depth/masks later
            "map1_l": map1_l,
            "map2_l": map2_l,
            "map1_r": map1_r,
            "map2_r": map2_r,
        }

    def left_to_right_from_cam2lidar(
        self,
        cam2lidar_left: np.ndarray,
        cam2lidar_right: np.ndarray,
    ):
        H_L = self._as_4x4(cam2lidar_left)
        H_R = self._as_4x4(cam2lidar_right)

        # Left -> Right via lidar:  p_R = inv(H_R) @ (H_L @ p_L)
        H_LR = np.linalg.inv(H_R) @ H_L

        R = H_LR[:3, :3].astype(np.float64, copy=True)
        t = H_LR[:3, 3].reshape(3, 1).astype(np.float64, copy=True)

        # Orthonormalize R
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        return R, t, H_LR

    def _as_4x4(self, H):
        H = np.asarray(H, dtype=np.float64)
        if H.shape == (4, 4):
            return H
        if H.shape == (3, 4):
            H4 = np.eye(4, dtype=np.float64)
            H4[:3, :] = H
            return H4
        raise ValueError(f"Expected (4,4) or (3,4) transform, got {H.shape}")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"gated_rect_loaded={self.rect_loaded_flags['gated']}, "
            f"rgb_rect_loaded={self.rect_loaded_flags['rgb']}, "
            f"rccb_rect_loaded={self.rect_loaded_flags['rccb']})"
        )

class GatedStereoScaleMultiViewImage(BaseTransform):
    """
    Per-camera-type (and optional per-view-tag) scaling for multi-view images.

    Supports:
      - results['img']: list of np.ndarray length N
      - results['view_names']: list of tags length N (e.g. 'gated_left', ...)
      - results['cam2img'], results['lidar2img'] as list or np.ndarray
      - optional: results['depth'] dict with lists 'std'/'mean'
      - optional: results['loaded_img_depth'] list
      - optional: results['gt_bboxes'] list
      - optional: results['camera'] tree to update intrinsics K
      - optional: results['sparse_depth'], results['gt_sparse_depth'] as list aligned with view_names
                  (sparse lidar depth, zeros invalid) -> rescaled via z-buffer splat (NOT cv2.resize)

    Scaling policy:
      - Provide target (H, W) per group/tag:
            scales = {
              "default": (448, 800),
              "gated":   (512, 1024),
              "rgb":     (448, 800),
              "rccb":    (448, 800),
              "gated_left": (512, 1024),   # optional override
            }
      - Each view gets its own scale matrix S (4x4):
            S[0,0] = sx = new_w / old_w
            S[1,1] = sy = new_h / old_h
        Then:
            cam2img[i]   <- S @ cam2img[i]
            lidar2img[i] <- S @ lidar2img[i]
    """

    TAG_TO_CAMERA = {
        "rccb_left": ("ar0820", "left"),
        "rccb_right": ("ar0820", "right"),
        "rgb_left": ("rgb", "left"),
        "rgb_right": ("rgb", "right"),
        "gated_left": ("gated", "left"),
        "gated_right": ("gated", "right"),
    }
    GATED_EXPOSURES = ["type8", "type9", "type10", "passive1", "passive2"]

    def __init__(
        self,
        scales: Optional[Dict[str, Tuple[int, int]]] = None,  # dict key->(H,W)
        rescale_radar: bool = False,
        rescale_depth: bool = True,
        update_camera_tree: bool = True,
        interpolation_img: Optional[int] = None,
        rescale_sparse_depth: bool = True,     # <= new: handle sparse_depth/gt_sparse_depth
        sparse_keys: Tuple[str, ...] = ("sparse_depth", "gt_sparse_depth"),
        sparse_rounding: str = "round",        # "round" or "floor"
        sparse_eps: float = 1e-12,
    ):
        super().__init__()

        if cv2 is None:
            raise ImportError("cv2 is required for GatedStereoScaleMultiViewImage")

        self.scales = scales or {"default": (448, 800)}
        self.rescale_radar = rescale_radar
        self.rescale_depth = rescale_depth
        self.update_camera_tree = update_camera_tree
        self.interpolation_img = cv2.INTER_LINEAR if interpolation_img is None else interpolation_img

        self.rescale_sparse_depth = bool(rescale_sparse_depth)
        self.sparse_keys = tuple(sparse_keys)
        assert sparse_rounding in ("round", "floor")
        self.sparse_rounding = sparse_rounding
        self.sparse_eps = float(sparse_eps)

        self.scales.setdefault("default", (448, 800))

    def _group_from_tag(self, tag: str) -> str:
        if not tag:
            return "default"
        return tag.split("_", 1)[0]

    def _resolve_target_hw(self, tag: str, filename: str = "") -> Tuple[int, int]:
        group = self._group_from_tag(tag)
        is_narrow = "narrow" in (filename or "")

        # priority: tag_narrow, group_narrow, tag, group, default
        if is_narrow and f"{tag}_narrow" in self.scales:
            return self.scales[f"{tag}_narrow"]
        if is_narrow and f"{group}_narrow" in self.scales:
            return self.scales[f"{group}_narrow"]
        if tag in self.scales:
            return self.scales[tag]
        if group in self.scales:
            return self.scales[group]
        return self.scales["default"]

    @staticmethod
    def _as_list(x, n: int):
        if x is None:
            return None
        if isinstance(x, list):
            return x
        if isinstance(x, np.ndarray):
            return [x[i] for i in range(n)]
        return x

    @staticmethod
    def _write_back(container, lst: list):
        if container is None:
            return None
        if isinstance(container, np.ndarray):
            try:
                return np.stack(lst, axis=0)
            except Exception:
                return lst
        return lst

    @staticmethod
    def _scale_matrix_4x4(sx: float, sy: float) -> np.ndarray:
        S = np.eye(4, dtype=np.float32)
        S[0, 0] = sx
        S[1, 1] = sy
        return S

    @staticmethod
    def _ensure_hw1(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            return arr[..., None]
        if arr.ndim == 3 and arr.shape[2] == 1:
            return arr
        return arr[..., :1]

    def _resize_sparse_depth_zbuffer(
        self,
        depth: np.ndarray,
        target_h: int,
        target_w: int,
        sx: float,
        sy: float,
    ) -> np.ndarray:
        """
        Rescale sparse depth where 0 means invalid by moving valid pixels and z-buffering.
        Returns (target_h, target_w, 1) float32.
        """
        d = self._ensure_hw1(np.asarray(depth))
        d2 = d[..., 0]

        mask = d2 > self.sparse_eps
        if not np.any(mask):
            return np.zeros((target_h, target_w, 1), dtype=np.float32)

        ys, xs = np.where(mask)
        z = d2[ys, xs].astype(np.float32)

        xs_f = xs.astype(np.float32) * float(sx)
        ys_f = ys.astype(np.float32) * float(sy)

        if self.sparse_rounding == "floor":
            x_new = np.floor(xs_f).astype(np.int32)
            y_new = np.floor(ys_f).astype(np.int32)
        else:
            x_new = np.rint(xs_f).astype(np.int32)
            y_new = np.rint(ys_f).astype(np.int32)

        valid = (x_new >= 0) & (x_new < target_w) & (y_new >= 0) & (y_new < target_h)
        if not np.any(valid):
            return np.zeros((target_h, target_w, 1), dtype=np.float32)

        x_new = x_new[valid]
        y_new = y_new[valid]
        z = z[valid]

        out = np.full((target_h, target_w), np.inf, dtype=np.float32)
        flat = (y_new.astype(np.int64) * int(target_w) + x_new.astype(np.int64))
        np.minimum.at(out.ravel(), flat, z)  # z-buffer: nearest point wins

        out[~np.isfinite(out)] = 0.0
        return out[..., None]

    def _update_camera_tree_K(self, results: dict, tag: str, sx: float, sy: float) -> None:
        if not self.update_camera_tree:
            return
        camera_tree = results.get("camera", None)
        if camera_tree is None:
            return
        if tag not in self.TAG_TO_CAMERA:
            return

        group_raw, side = self.TAG_TO_CAMERA[tag]

        def _scale_K(intr: dict):
            if intr is None:
                return
            K = intr.get("K", None)
            if K is None:
                return
            K = np.asarray(K, dtype=np.float64)
            if K.shape != (3, 3):
                return
            K[0, 0] *= float(sx)
            K[0, 2] *= float(sx)
            K[1, 1] *= float(sy)
            K[1, 2] *= float(sy)
            intr["K"] = K.tolist()

        if group_raw == "gated":
            side_node = camera_tree.get("gated", {}).get(side, {})
            for exp in self.GATED_EXPOSURES:
                if exp in side_node:
                    intr = side_node[exp].get("intrinsics", None)
                    _scale_K(intr)
        else:
            node = camera_tree.get(group_raw, {}).get(side, None)
            if node is None:
                return
            intr = node.get("intrinsics", None)
            _scale_K(intr)

    def transform(self, results: dict):
        if "img" not in results:
            return results

        imgs = results["img"]
        if not isinstance(imgs, list):
            raise TypeError("results['img'] must be a list of per-view images")

        n = len(imgs)
        view_names = results.get("view_names", [""] * n)
        filenames = results.get("filename", [""] * n)
        if isinstance(filenames, str):
            filenames = [filenames]

        cam2img_l = self._as_list(results.get("cam2img", None), n)
        lidar2img_l = self._as_list(results.get("lidar2img", None), n)
        lidar2cam_l = self._as_list(results.get("lidar2cam", None), n)

        depth_std = None
        depth_mean = None
        if "depth" in results and isinstance(results["depth"], dict):
            depth_std = results["depth"].get("std", None)
            depth_mean = results["depth"].get("mean", None)

        loaded_img_depth = results.get("loaded_img_depth", None)
        if loaded_img_depth is not None and not isinstance(loaded_img_depth, list):
            raise TypeError("loaded_img_depth must be a list (multi-view)")

        gt_bboxes = results.get("gt_bboxes", None)

        # sparse depth containers (lists aligned with view_names)
        sparse_lists = {}
        if self.rescale_sparse_depth and self.rescale_depth:
            for k in self.sparse_keys:
                v = results.get(k, None)
                if isinstance(v, list) and len(v) == n:
                    sparse_lists[k] = v

        new_imgs = []
        scale_factors_4x4 = []

        for i in range(n):
            tag = view_names[i] if i < len(view_names) else ""
            fn = filenames[i] if i < len(filenames) else ""
            target_h, target_w = self._resolve_target_hw(tag, fn)

            old_h, old_w = imgs[i].shape[:2]
            if old_h <= 0 or old_w <= 0:
                raise ValueError(f"Invalid image shape for view {i}: {imgs[i].shape}")

            sx = float(target_w) / float(old_w)
            sy = float(target_h) / float(old_h)

            # --- resize image (preserve (H,W,1) as (H,W,1)) ---
            orig = imgs[i]
            is_1ch_3d = (orig.ndim == 3 and orig.shape[2] == 1)

            if (old_h, old_w) != (target_h, target_w):
                if is_1ch_3d:
                    tmp = orig[..., 0]
                    tmp2 = cv2.resize(tmp, (int(target_w), int(target_h)), interpolation=self.interpolation_img)
                    new_img = tmp2[..., None]
                else:
                    new_img = cv2.resize(orig, (int(target_w), int(target_h)), interpolation=self.interpolation_img)
                    if new_img.ndim == 2:
                        new_img = new_img[..., None]
            else:
                new_img = orig
            new_imgs.append(new_img)

            S = self._scale_matrix_4x4(sx, sy)
            scale_factors_4x4.append(S)

            # --- resize dense depth stats (NEAREST) ---
            if depth_std is not None and isinstance(depth_std, list) and i < len(depth_std) and depth_std[i] is not None:
                d = depth_std[i]
                d2 = cv2.resize(
                    d.squeeze() if d.ndim == 3 and d.shape[2] == 1 else d,
                    (int(target_w), int(target_h)),
                    interpolation=cv2.INTER_NEAREST,
                )
                depth_std[i] = d2[..., None] if d2.ndim == 2 else d2

            if depth_mean is not None and isinstance(depth_mean, list) and i < len(depth_mean) and depth_mean[i] is not None:
                d = depth_mean[i]
                d2 = cv2.resize(
                    d.squeeze() if d.ndim == 3 and d.shape[2] == 1 else d,
                    (int(target_w), int(target_h)),
                    interpolation=cv2.INTER_NEAREST,
                )
                depth_mean[i] = d2[..., None] if d2.ndim == 2 else d2

            # --- resize loaded_img_depth (NEAREST) preserving channel ---
            if loaded_img_depth is not None and self.rescale_depth and i < len(loaded_img_depth) and loaded_img_depth[i] is not None:
                d0 = loaded_img_depth[i]
                is_1ch_3d = (d0.ndim == 3 and d0.shape[2] == 1)
                if is_1ch_3d:
                    tmp = d0[..., 0]
                    tmp2 = cv2.resize(tmp, (int(target_w), int(target_h)), interpolation=cv2.INTER_NEAREST)
                    loaded_img_depth[i] = tmp2[..., None]
                else:
                    d2 = cv2.resize(d0, (int(target_w), int(target_h)), interpolation=cv2.INTER_NEAREST)
                    if d2.ndim == 2:
                        d2 = d2[..., None]
                    loaded_img_depth[i] = d2

            # --- rescale sparse depth maps (z-buffer splat, zeros invalid) ---
            if sparse_lists:
                for k, dlist in sparse_lists.items():
                    if i < len(dlist) and dlist[i] is not None:
                        dlist[i] = self._resize_sparse_depth_zbuffer(
                            dlist[i],
                            target_h=int(target_h),
                            target_w=int(target_w),
                            sx=sx,
                            sy=sy,
                        )

            # --- scale 2D bboxes ---
            if gt_bboxes is not None:
                if isinstance(gt_bboxes, list):
                    if i < len(gt_bboxes) and gt_bboxes[i] is not None:
                        box = gt_bboxes[i]
                        box[:, [0, 2]] *= sx
                        box[:, [1, 3]] *= sy
                        gt_bboxes[i] = box
                else:
                    gt_bboxes[:, [0, 2]] *= sx
                    gt_bboxes[:, [1, 3]] *= sy

            # --- update camera tree intrinsics K ---
            self._update_camera_tree_K(results, tag, sx, sy)

        results["img"] = new_imgs
        results["img_shape"] = [img.shape for img in results["img"]]
        results["scale_factor"] = [scale_factors_4x4[i] for i in range(n)]

        if loaded_img_depth is not None:
            results["loaded_img_depth"] = loaded_img_depth
        if gt_bboxes is not None:
            results["gt_bboxes"] = gt_bboxes
        if depth_std is not None or depth_mean is not None:
            results.setdefault("depth", {})
            if depth_std is not None:
                results["depth"]["std"] = depth_std
            if depth_mean is not None:
                results["depth"]["mean"] = depth_mean

        # write sparse depth lists back
        for k, v in sparse_lists.items():
            results[k] = v

        # update projections
        if cam2img_l is not None:
            for i in range(n):
                if cam2img_l[i] is not None:
                    cam2img_l[i] = scale_factors_4x4[i] @ cam2img_l[i]
            results["cam2img"] = self._write_back(results.get("cam2img", None), cam2img_l)

        if lidar2img_l is not None:
            for i in range(n):
                if lidar2img_l[i] is not None:
                    lidar2img_l[i] = scale_factors_4x4[i] @ lidar2img_l[i]
            results["lidar2img"] = self._write_back(results.get("lidar2img", None), lidar2img_l)
        else:
            # if lidar2img absent but lidar2cam present, recompute if cam2img exists
            if ("cam2img" in results) and (lidar2cam_l is not None):
                c2i = self._as_list(results["cam2img"], n)
                out = []
                for i in range(n):
                    out.append(None if (c2i[i] is None or lidar2cam_l[i] is None) else (c2i[i] @ lidar2cam_l[i]))
                results["lidar2img"] = out

        return results

    def __repr__(self):
        keys = ", ".join(sorted(self.scales.keys()))
        return f"{self.__class__.__name__}(scales_keys=[{keys}], rescale_depth={self.rescale_depth}, rescale_sparse_depth={self.rescale_sparse_depth})"

class GatedStereoCropMultiViewImages(BaseTransform):
    """
    Per-camera-type (and optional per-view-tag) cropping for multi-view images.

    Expected:
      - results['img']: List[np.ndarray] length N
      - results['view_names']: List[str] length N (e.g. 'gated_left', 'rgb_right', ...)
      - results['cam2img'], results['lidar2cam'] as either:
            * np.ndarray of shape (N,4,4) or (N,3,4) / (N,3,3)
            * or list of per-view matrices
      - optionally: results['cam2lidar'], results['lidar2img']
      - optionally: results['camera'] tree for writing back intrinsics/cam2lidar

    Also supports sparse depth maps (zeros invalid) aligned with view_names:
      - results['sparse_depth'], results['gt_sparse_depth']  (list length N)
        Cropped via array slicing (keeps zeros invalid).

    Crops can be specified by:
      1) exact view tag key, e.g. crops['gated_left']
      2) group key, e.g. crops['gated'], crops['rgb'], crops['rccb']
      3) 'default'

    Optional narrow override:
      - if 'narrow' is in filename for a view, and crops has key '{group}_narrow'
        or '{tag}_narrow', that config is used.
    """

    TAG_TO_CAMERA = {
        "rccb_left": ("ar0820", "left"),
        "rccb_right": ("ar0820", "right"),
        "rgb_left": ("rgb", "left"),
        "rgb_right": ("rgb", "right"),
        "gated_left": ("gated", "left"),
        "gated_right": ("gated", "right"),
    }
    GATED_EXPOSURES = ["type8", "type9", "type10", "passive1", "passive2"]

    def __init__(
        self,
        crops: Optional[Dict[str, Dict[str, int]]] = None,
        update_camera_tree: bool = True,
        also_crop_depth_fields: bool = True,
        also_shift_gt_bboxes: bool = True,
        also_crop_sparse_depth: bool = True,
        sparse_keys: tuple = ("sparse_depth", "gt_sparse_depth"),
    ) -> None:
        super().__init__()
        self.crops = crops or {}
        self.update_camera_tree = update_camera_tree
        self.also_crop_depth_fields = also_crop_depth_fields
        self.also_shift_gt_bboxes = also_shift_gt_bboxes
        self.also_crop_sparse_depth = bool(also_crop_sparse_depth)
        self.sparse_keys = tuple(sparse_keys)

        self.crops.setdefault("default", dict(from_top=0, from_bottom=0, from_left=0, from_right=0))

    # ------------------------------
    # core helpers
    # ------------------------------
    @staticmethod
    def _crop(arr: np.ndarray, bbox_xyxy_inclusive: np.ndarray) -> np.ndarray:
        """
        Crops any array shaped (H,W), (H,W,1), (H,W,C).
        Preserves trailing channel dim if present.
        """
        x1, y1, x2, y2 = map(int, bbox_xyxy_inclusive)
        h, w = arr.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if arr.ndim == 2:
            return arr[y1:y2 + 1, x1:x2 + 1]
        return arr[y1:y2 + 1, x1:x2 + 1, ...]

    @staticmethod
    def _as_4x4(H: Any) -> np.ndarray:
        H = np.asarray(H, dtype=np.float64)
        if H.shape == (4, 4):
            return H
        if H.shape == (3, 4):
            out = np.eye(4, dtype=np.float64)
            out[:3, :] = H
            return out
        if H.shape == (3, 3):
            out = np.eye(4, dtype=np.float64)
            out[:3, :3] = H
            return out
        raise ValueError(f"Expected (4,4), (3,4) or (3,3), got {H.shape}")

    @staticmethod
    def _inv4x4(H: Any) -> np.ndarray:
        return np.linalg.inv(GatedStereoCropMultiViewImages._as_4x4(H))

    def _group_from_tag(self, tag: str) -> str:
        if not tag:
            return "default"
        return tag.split("_", 1)[0]

    def _resolve_crop_cfg(self, tag: str, filename: str) -> Dict[str, int]:
        group = self._group_from_tag(tag)
        is_narrow = ("narrow" in (filename or ""))

        if is_narrow and f"{tag}_narrow" in self.crops:
            cfg = self.crops[f"{tag}_narrow"]
        elif is_narrow and f"{group}_narrow" in self.crops:
            cfg = self.crops[f"{group}_narrow"]
        elif tag in self.crops:
            cfg = self.crops[tag]
        elif group in self.crops:
            cfg = self.crops[group]
        else:
            cfg = self.crops["default"]

        return dict(
            from_top=int(cfg.get("from_top", 0)),
            from_bottom=int(cfg.get("from_bottom", 0)),
            from_left=int(cfg.get("from_left", 0)),
            from_right=int(cfg.get("from_right", 0)),
        )

    def _shift_cam2img_in_place(self, cam2img_i: Any, crop_left: int, crop_top: int) -> None:
        if cam2img_i is None:
            return
        cam2img_i = np.asarray(cam2img_i)
        cam2img_i[0, 2] -= float(crop_left)
        cam2img_i[1, 2] -= float(crop_top)

    def _recompute_derived_mats(self, results: dict) -> None:
        # cam2lidar from lidar2cam if present
        if "lidar2cam" in results:
            l2c = results["lidar2cam"]
            if isinstance(l2c, np.ndarray):
                cam2lidar = np.stack([self._inv4x4(l2c[i]) for i in range(l2c.shape[0])], axis=0)
            else:
                cam2lidar = [self._inv4x4(m) if m is not None else None for m in l2c]
            results["cam2lidar"] = cam2lidar

        # lidar2img = cam2img @ lidar2cam if both present
        if "cam2img" in results and "lidar2cam" in results:
            c2i = results["cam2img"]
            l2c = results["lidar2cam"]

            if isinstance(c2i, np.ndarray) and isinstance(l2c, np.ndarray):
                results["lidar2img"] = c2i @ l2c
            else:
                out = []
                n = len(results.get("img", []))
                for i in range(n):
                    c = c2i[i] if isinstance(c2i, list) else c2i[i, ...]
                    l = l2c[i] if isinstance(l2c, list) else l2c[i, ...]
                    out.append(None if (c is None or l is None) else (c @ l))
                results["lidar2img"] = out

    def _update_camera_tree_for_view(self, results: dict, tag: str, crop_left: int, crop_top: int) -> None:
        if not self.update_camera_tree:
            return
        camera_tree = results.get("camera", None)
        if camera_tree is None:
            return
        if "view_names" not in results or tag not in results["view_names"]:
            return

        idx = results["view_names"].index(tag)
        cam2lidar_arr = results.get("cam2lidar", None)

        if tag not in self.TAG_TO_CAMERA:
            return
        group_raw, side = self.TAG_TO_CAMERA[tag]

        cam2lidar_new = None
        if cam2lidar_arr is not None:
            if isinstance(cam2lidar_arr, np.ndarray):
                cam2lidar_new = cam2lidar_arr[idx].tolist()
            else:
                cam2lidar_new = cam2lidar_arr[idx].tolist() if cam2lidar_arr[idx] is not None else None

        def _shift_K_dict(intr_node: dict):
            if intr_node is None:
                return
            K = intr_node.get("K", None)
            if K is None:
                return
            K = np.asarray(K, dtype=np.float64)
            if K.shape != (3, 3):
                return
            K[0, 2] -= float(crop_left)
            K[1, 2] -= float(crop_top)
            intr_node["K"] = K.tolist()

        if group_raw == "gated":
            side_node = camera_tree.get("gated", {}).get(side, {})
            for exp in self.GATED_EXPOSURES:
                if exp in side_node:
                    if cam2lidar_new is not None:
                        side_node[exp]["cam2lidar"] = cam2lidar_new
                    intr = side_node[exp].get("intrinsics", None)
                    _shift_K_dict(intr)
        else:
            node = camera_tree.get(group_raw, {}).get(side, None)
            if node is None:
                return
            if cam2lidar_new is not None:
                node["cam2lidar"] = cam2lidar_new
            intr = node.get("intrinsics", None)
            _shift_K_dict(intr)

    # ------------------------------
    # main API
    # ------------------------------
    def transform(self, results: dict) -> Optional[dict]:
        if "img" not in results:
            return results

        imgs = results["img"]
        if not isinstance(imgs, list):
            raise TypeError("results['img'] must be a list of per-view images.")

        n = len(imgs)
        view_names = results.get("view_names", [""] * n)
        filenames = results.get("filename", [""] * n)
        if isinstance(filenames, str):
            filenames = [filenames]

        gt_bboxes = results.get("gt_bboxes", None)
        if gt_bboxes is not None and not isinstance(gt_bboxes, list):
            raise TypeError("gt_bboxes must be a list (multi-view).")

        loaded_img_depths = results.get("loaded_img_depth", None)
        if loaded_img_depths is not None and not isinstance(loaded_img_depths, list):
            raise TypeError("loaded_img_depth must be a list (multi-view).")

        cam2imgs = results.get("cam2img", None)
        lidar2cams = results.get("lidar2cam", None)

        # sparse depth lists aligned with views (zeros invalid -> cropping is safe)
        sparse_lists: Dict[str, list] = {}
        if self.also_crop_sparse_depth:
            for k in self.sparse_keys:
                v = results.get(k, None)
                if isinstance(v, list) and len(v) == n:
                    sparse_lists[k] = v

        new_shapes_hw = []

        for i in range(n):
            tag = view_names[i] if i < len(view_names) else ""
            fn = filenames[i] if i < len(filenames) else ""
            cfg = self._resolve_crop_cfg(tag, fn)

            crop_top = cfg["from_top"]
            crop_bottom = cfg["from_bottom"]
            crop_left = cfg["from_left"]
            crop_right = cfg["from_right"]

            h, w = imgs[i].shape[:2]
            new_w = w - crop_left - crop_right
            new_h = h - crop_top - crop_bottom
            if new_w <= 0 or new_h <= 0:
                raise ValueError(
                    f"[{self.__class__.__name__}] invalid crop for view {i} ({tag}): "
                    f"orig=({h},{w}), crop LRTB=({crop_left},{crop_right},{crop_top},{crop_bottom}) -> "
                    f"new=({new_h},{new_w})"
                )

            x1, y1 = crop_left, crop_top
            x2, y2 = x1 + new_w - 1, y1 + new_h - 1
            bbox = np.array([x1, y1, x2, y2], dtype=np.int32)

            # crop image
            imgs[i] = self._crop(imgs[i], bbox)

            # crop optional depth dict fields (std/mean)
            if self.also_crop_depth_fields and isinstance(results.get("depth", None), dict):
                d = results["depth"]
                for key in ("std", "mean"):
                    if key in d and isinstance(d[key], list) and i < len(d[key]) and d[key][i] is not None:
                        d[key][i] = self._crop(d[key][i], bbox)

            # crop optional per-view dense depth arrays
            if loaded_img_depths is not None and i < len(loaded_img_depths) and loaded_img_depths[i] is not None:
                loaded_img_depths[i] = self._crop(loaded_img_depths[i], bbox)

            # crop sparse depth maps (keeps zeros invalid)
            for k, dlist in sparse_lists.items():
                if i < len(dlist) and dlist[i] is not None:
                    dlist[i] = self._crop(dlist[i], bbox)

            # shift bboxes if requested
            if self.also_shift_gt_bboxes and gt_bboxes is not None and i < len(gt_bboxes) and gt_bboxes[i] is not None:
                boxes_i = gt_bboxes[i]
                boxes_i[:, [0, 2]] -= crop_left
                boxes_i[:, [1, 3]] -= crop_top
                gt_bboxes[i] = boxes_i

            # update cam2img principal point (cx, cy)
            if cam2imgs is not None:
                if isinstance(cam2imgs, np.ndarray):
                    self._shift_cam2img_in_place(cam2imgs[i], crop_left, crop_top)
                else:
                    self._shift_cam2img_in_place(cam2imgs[i], crop_left, crop_top)

            # update camera tree intrinsics K + cam2lidar consistency
            self._update_camera_tree_for_view(results, tag, crop_left, crop_top)

            new_shapes_hw.append(imgs[i].shape[:2])

        # write back
        results["img"] = imgs
        if loaded_img_depths is not None:
            results["loaded_img_depth"] = loaded_img_depths
        if gt_bboxes is not None:
            results["gt_bboxes"] = gt_bboxes
        for k, v in sparse_lists.items():
            results[k] = v

        if cam2imgs is not None:
            results["cam2img"] = cam2imgs
        if lidar2cams is not None:
            results["lidar2cam"] = lidar2cams

        self._recompute_derived_mats(results)

        # shape bookkeeping
        all_same = len(set(new_shapes_hw)) == 1
        if all_same:
            H, W = new_shapes_hw[0]
            results["img_shape"] = (H, W)
            results["ori_shape"] = (H, W)
            results["pad_shape"] = (H, W)
        else:
            results["img_shape"] = [img.shape for img in imgs]
            results["ori_shape"] = [img.shape for img in imgs]
            results["pad_shape"] = [img.shape for img in imgs]

        return results

    def __repr__(self) -> str:
        keys = ", ".join(sorted(self.crops.keys()))
        return (
            f"{self.__class__.__name__}("
            f"crops_keys=[{keys}], "
            f"update_camera_tree={self.update_camera_tree}, "
            f"also_crop_sparse_depth={self.also_crop_sparse_depth})"
        )


def filter_occluded_points_lidar_depth(depth, radius=3, thr=1.25):
    H, W, N = depth.shape
    # Initialize depth buffer and sort out mask for all depth maps
    depth_buffer = np.full_like(depth, np.inf, dtype=np.float32)
    sort_out_mask = np.ones_like(depth, dtype=np.uint8)

    # Create a structuring element for the circular mask
    kernel_size = radius * 2 + 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
    cv2.circle(kernel, (radius, radius), radius, 1, -1)

    # Get positions of valid depth values across all maps and sort them
    pos = np.column_stack(np.where(depth > 0))  # Shape: (M, 3), where M is the total number of valid positions
    depth_values = depth[depth > 0]
    sorted_indices = np.argsort(depth_values)
    sorted_pos = pos[sorted_indices]
    sorted_depth_values = depth_values[sorted_indices]

    # Main loop over sorted depth values (in parallel for all depth maps)
    for idx in range(len(sorted_pos)):
        y, x, z = sorted_pos[idx]
        color = sorted_depth_values[idx]

        # Define the region of interest
        y0, y1 = max(0, y - radius), min(H, y + radius + 1)
        x0, x1 = max(0, x - radius), min(W, x + radius + 1)

        # Extract the relevant portion of the depth buffer and kernel across all maps
        roi_depth_buffer = depth_buffer[y0:y1, x0:x1, z]
        roi_kernel = kernel[(y0 - y + radius):(y1 - y + radius), (x0 - x + radius):(x1 - x + radius)]

        # Find the minimum depth in the overlapping region (across all depth maps)
        overlapping_depths = roi_depth_buffer[roi_kernel == 1]
        min_overlapping_depth = overlapping_depths.min() if overlapping_depths.size > 0 else np.inf

        # Occlusion check for all maps simultaneously
        if color / min_overlapping_depth > thr:
            sort_out_mask[y, x, z] = 0
        else:
            # Update depth buffer in the masked region
            roi_depth_buffer[roi_kernel == 1] = color

    # Resulting mask for non-occluded points across all depth maps
    non_occluded_points = sort_out_mask == 1

    return non_occluded_points
class GenerateSparseDepth(BaseTransform):
    """
    Produces:
      - results["gt_sparse_depth"]: list of per-view depth maps from FULL point cloud
      - results["sparse_depth"]:    list of per-view depth maps from SUBSAMPLED point cloud

    If and only if results["img"] contains multiple view resolutions:
      - results["gt_sparse_depth_hd"]: list of per-view depth maps at GLOBAL max (H,W) across views
      - results["sparse_depth_hd"]:    list of per-view depth maps at GLOBAL max (H,W) across views

    Occlusion filtering:
      - occlusion mask is computed from FULL depth map only
      - sparse depth is multiplied by that same mask (no separate mask computed)

    Supports different view resolutions / channels automatically.
    """

    def __init__(
        self,
        res_depth=None,
        filter_occluded_points: Optional[bool] = None,
        use_accumulated_points: bool = False,
        random_mask_gt_points: bool = False,
        sparse_num_points: Optional[int] = None,
        sparse_ratio: Optional[float] = None,
        eps: float = 1e-5,
        add_pixel_center_offset: bool = False,
    ):
        super().__init__()
        self.res_depth = res_depth
        self.filter_occluded_points = filter_occluded_points
        self.use_accumulated_points = use_accumulated_points
        self.random_mask_gt_points = random_mask_gt_points
        self.sparse_num_points = sparse_num_points
        self.sparse_ratio = sparse_ratio
        self.eps = float(eps)
        self.add_pixel_center_offset = bool(add_pixel_center_offset)

    def transform(self, results: dict) -> Optional[dict]:
        if "img" not in results or "lidar2img" not in results:
            return results

        nviews = len(results["img"])
        if nviews == 0:
            results["gt_sparse_depth"] = []
            results["sparse_depth"] = []
            results["gt_sparse_depth_hd"] = None
            results["sparse_depth_hd"] = None
            return results

        # -------- determine if resolutions differ across views --------
        view_hw = []
        for i in range(nviews):
            img_i = results["img"][i]
            H_i, W_i = img_i.shape[:2]
            view_hw.append((int(H_i), int(W_i)))

        unique_hw = set(view_hw)
        do_hd = (len(unique_hw) > 1)

        max_H = max(h for h, _ in view_hw)
        max_W = max(w for _, w in view_hw)

        # HD depth resolution uses the same res_depth logic, but anchored at max image size.
        depth_H_hd, depth_W_hd = self._depth_hw(max_H, max_W) if do_hd else (None, None)

        # -------- points selection (full) --------
        points_full = self._get_points_full(results)

        if points_full is None or len(points_full) == 0:
            # no points, produce empty depth maps
            results["gt_sparse_depth"] = [None] * nviews
            results["sparse_depth"] = [None] * nviews
            results["gt_sparse_depth_hd"] = ([None] * nviews) if do_hd else None
            results["sparse_depth_hd"] = ([None] * nviews) if do_hd else None
            if self.filter_occluded_points:
                results["occlusion_mask"] = [None] * nviews
            return results

        # -------- points selection (sparse) --------
        aug = results.get("augmentations", {}) if isinstance(results.get("augmentations", {}), dict) else {}
        sparse_ratio_override = aug.get("sparse_ratio", None)
        sparse_num_points_override = aug.get("sparse_num_points", None)

        points_sparse = self._subsample_points(
            points_full,
            sparse_ratio=sparse_ratio_override,
            sparse_num_points=sparse_num_points_override,
        )

        # -------- per-view projection --------
        lidar2img_sparse = np.asarray(results["lidar2img"], dtype=np.float64)
        lidar2img_gt = results.get("lidar2img_gt", None)
        if lidar2img_gt is None:
            lidar2img_gt = lidar2img_sparse
        else:
            lidar2img_gt = np.asarray(lidar2img_gt, dtype=np.float64)

        gt_depth_list = []
        sparse_depth_list = []
        occ_mask_list = []

        gt_depth_list_hd = None
        sparse_depth_list_hd = None
        occ_mask_list_hd = None
        if do_hd:
            gt_depth_list_hd = []
            sparse_depth_list_hd = []
            occ_mask_list_hd = []

        # 1) build GT depth (FULL cloud) + occlusion mask (normal + hd)
        for i in range(nviews):
            img_i = results["img"][i]
            ori_H, ori_W = img_i.shape[:2]
            depth_H, depth_W = self._depth_hw(ori_H, ori_W)

            depth_full = self._project_to_depth(
                points_full,
                lidar2img_gt[i],
                ori_H=ori_H,
                ori_W=ori_W,
                depth_H=depth_H,
                depth_W=depth_W,
            )  # (H,W,1)

            if self.filter_occluded_points:
                mask = filter_occluded_points_lidar_depth(depth_full)
                depth_full = depth_full * mask
                occ_mask_list.append(mask)
            else:
                occ_mask_list.append(None)

            gt_depth_list.append(depth_full)

            # HD version (global max resolution)
            if do_hd:
                depth_full_hd = self._project_to_depth(
                    points_full,
                    lidar2img_gt[i],
                    ori_H=ori_H,
                    ori_W=ori_W,
                    depth_H=depth_H_hd,
                    depth_W=depth_W_hd,
                )

                if self.filter_occluded_points:
                    mask_hd = filter_occluded_points_lidar_depth(depth_full_hd)
                    depth_full_hd = depth_full_hd * mask_hd
                    occ_mask_list_hd.append(mask_hd)
                else:
                    occ_mask_list_hd.append(None)

                gt_depth_list_hd.append(depth_full_hd)

        # optional: "points_low_range" fill holes in GT only (normal + hd)
        if results.get("points_low_range", None) is not None:
            points_lr = self._to_numpy(results["points_low_range"])
            if points_lr is not None and len(points_lr) > 0:
                for i in range(nviews):
                    img_i = results["img"][i]
                    ori_H, ori_W = img_i.shape[:2]
                    depth_H, depth_W = self._depth_hw(ori_H, ori_W)

                    depth_lr = self._project_to_depth(
                        points_lr,
                        lidar2img_gt[i],
                        ori_H=ori_H,
                        ori_W=ori_W,
                        depth_H=depth_H,
                        depth_W=depth_W,
                    )

                    gt = gt_depth_list[i]
                    fill = (gt == 0)
                    gt_depth_list[i] = gt + depth_lr * fill

                    if do_hd:
                        depth_lr_hd = self._project_to_depth(
                            points_lr,
                            lidar2img_gt[i],
                            ori_H=ori_H,
                            ori_W=ori_W,
                            depth_H=depth_H_hd,
                            depth_W=depth_W_hd,
                        )
                        gt_hd = gt_depth_list_hd[i]
                        fill_hd = (gt_hd == 0)
                        gt_depth_list_hd[i] = gt_hd + depth_lr_hd * fill_hd

        # 2) build SPARSE depth (SUBSAMPLED cloud), reusing corresponding occlusion mask
        for i in range(nviews):
            img_i = results["img"][i]
            ori_H, ori_W = img_i.shape[:2]
            depth_H, depth_W = self._depth_hw(ori_H, ori_W)

            depth_sparse = self._project_to_depth(
                points_sparse,
                lidar2img_sparse[i],
                ori_H=ori_H,
                ori_W=ori_W,
                depth_H=depth_H,
                depth_W=depth_W,
            )

            if self.filter_occluded_points:
                mask = occ_mask_list[i]
                if mask is not None:
                    depth_sparse = depth_sparse * mask

            sparse_depth_list.append(depth_sparse)

            if do_hd:
                depth_sparse_hd = self._project_to_depth(
                    points_sparse,
                    lidar2img_sparse[i],
                    ori_H=ori_H,
                    ori_W=ori_W,
                    depth_H=depth_H_hd,
                    depth_W=depth_W_hd,
                )

                if self.filter_occluded_points:
                    mask_hd = occ_mask_list_hd[i]
                    if mask_hd is not None:
                        depth_sparse_hd = depth_sparse_hd * mask_hd

                sparse_depth_list_hd.append(depth_sparse_hd)

        # -------- store outputs --------
        results["gt_sparse_depth"] = gt_depth_list
        results["sparse_depth"] = sparse_depth_list

        if do_hd:
            results["gt_sparse_depth_hd"] = gt_depth_list_hd
            results["sparse_depth_hd"] = sparse_depth_list_hd

        if self.filter_occluded_points:
            results["occlusion_mask"] = occ_mask_list
            # If you want this too, uncomment:
            # results["occlusion_mask_hd"] = occ_mask_list_hd if do_hd else None

        return results

    # ---------------- helpers ----------------

    def _to_numpy(self, x):
        if x is None:
            return None
        if torch is not None and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _get_points_full(self, results: dict) -> Optional[np.ndarray]:
        """
        Full cloud for GT projection.
        Priority:
          - if use_accumulated_points and 'points_accumulated' exists: use that
          - else: use 'gt_points' if present else 'points'
        """
        if self.use_accumulated_points and results.get("points_accumulated", None) is not None:
            pts = results["points_accumulated"]
        else:
            pts = results.get("gt_points", None)
            if pts is None:
                pts = results.get("points", None)
        pts = self._to_numpy(pts)
        if pts is None:
            return None
        pts = pts.reshape(-1, pts.shape[-1])
        return pts

    def _subsample_points(
        self,
        points_full: np.ndarray,
        sparse_num_points: int = None,
        sparse_ratio: float = None,
        allow_empty: bool = True,
    ) -> np.ndarray:
        """Return subsampled point cloud used for sparse depth.

        Precedence:
          1) function args sparse_num_points / sparse_ratio (if not None)
          2) self.sparse_num_points / self.sparse_ratio (if not None)
          3) self.random_mask_gt_points
          4) else: no subsampling (sparse == gt)
        """
        pts = points_full
        n = int(pts.shape[0])
        if n == 0:
            return pts

        num_points_cfg = sparse_num_points if sparse_num_points is not None else self.sparse_num_points
        ratio_cfg = sparse_ratio if sparse_ratio is not None else self.sparse_ratio

        k = None

        if num_points_cfg is not None:
            k = int(num_points_cfg)
            k = max(0 if allow_empty else 1, min(n, k))

        elif ratio_cfg is not None:
            r = float(ratio_cfg)
            r = max(0.0, min(1.0, r))
            k = int(np.floor(n * r))
            k = max(0 if allow_empty else 1, min(n, k))

        elif self.random_mask_gt_points:
            r = float(np.random.rand() ** 3)
            k = int(np.floor(n * r))
            k = max(0 if allow_empty else 1, min(n, k))

        else:
            return pts

        if k is None:
            return pts
        if k <= 0:
            return pts[:0]
        if k >= n:
            return pts

        idx = np.random.choice(n, size=k, replace=False)
        return pts[idx]

    def _depth_hw(self, ori_H: int, ori_W: int) -> Tuple[int, int]:
        """Compute output depth map resolution per view."""
        if self.res_depth is None:
            return int(ori_H), int(ori_W)
        if isinstance(self.res_depth, float):
            return int(ori_H * self.res_depth), int(ori_W * self.res_depth)
        if isinstance(self.res_depth, (tuple, list)) and len(self.res_depth) == 2:
            return int(self.res_depth[0]), int(self.res_depth[1])
        raise ValueError("res_depth must be None, float, or (H,W) tuple")

    def _project_to_depth(
        self,
        points: np.ndarray,
        lidar2img_i: np.ndarray,
        ori_H: int,
        ori_W: int,
        depth_H: int,
        depth_W: int,
    ) -> np.ndarray:
        """Project points into a single view and produce a depth map HxWx1.
        Uses z-buffer (min depth) per pixel for stability.

        points: (N, >=3)
        lidar2img_i: (4,4)
        """
        eps = self.eps

        xyz = points[:, :3].astype(np.float64, copy=False)
        ones = np.ones((xyz.shape[0], 1), dtype=np.float64)
        pts_h = np.concatenate([xyz, ones], axis=1)  # (N,4)

        proj = pts_h @ np.asarray(lidar2img_i, dtype=np.float64).T

        z = proj[:, 2]
        z_safe = np.maximum(z, eps)

        u = proj[:, 0] / z_safe
        v = proj[:, 1] / z_safe
        if self.add_pixel_center_offset:
            u = u + 0.5
            v = v + 0.5

        valid = (z > eps) & (u > 0) & (u < ori_W) & (v > 0) & (v < ori_H)
        if not np.any(valid):
            return np.zeros((depth_H, depth_W, 1), dtype=np.float32)

        u = u[valid]
        v = v[valid]
        z = z[valid]

        u_idx = (u / float(ori_W) * float(depth_W)).astype(np.int32)
        v_idx = (v / float(ori_H) * float(depth_H)).astype(np.int32)

        u_idx = np.clip(u_idx, 0, depth_W - 1)
        v_idx = np.clip(v_idx, 0, depth_H - 1)

        depth = np.full((depth_H, depth_W), np.inf, dtype=np.float32)
        flat = (v_idx.astype(np.int64) * depth_W + u_idx.astype(np.int64))
        np.minimum.at(depth.ravel(), flat, z.astype(np.float32))

        depth[~np.isfinite(depth)] = 0.0
        return depth[..., None]

class CalculateCam2CamExtrinsics(BaseTransform):
    """
    Builds camera-to-camera extrinsics (nested dict):
        results[store_key][src_tag][dst_tag] = T_src_to_dst   (4x4)

    Mode A (preferred):
      - results['cam2lidar'] or results['lidar2cam'] aligned with view_names

    Mode B (MS2 fallback):
      - results['extrinsics'] as flat cross-modality LEFT<->LEFT dict (e.g. NIR2RGB, RGB2THR, ...)
      - results['stereo'][mod]['extrinsics_L2R'] as LEFT->RIGHT for each mod
      - view_names follow "{mod}_left"/"{mod}_right"

    In fallback, we build T_refLeft->view for all views and then:
        T_src_to_dst = T_ref->dst @ inv(T_ref->src)
    """

    def __init__(
        self,
        store_key: str = "cam2cam_extrinsics",  # <= strongly recommend NOT "extrinsics" for MS2
        dtype: str = "float64",
        include_identity: bool = True,
        ms2_reference_mod: str = "nir",         # MS2 calib cross-extrinsics are defined from NIR-left
    ):
        super().__init__()
        self.store_key = store_key
        self.dtype = np.float64 if dtype == "float64" else np.float32
        self.include_identity = include_identity
        self.ms2_reference_mod = ms2_reference_mod.lower()

    @staticmethod
    def _as_4x4(H: Any) -> np.ndarray:
        H = np.asarray(H, dtype=np.float64)
        if H.shape == (4, 4):
            return H
        if H.shape == (3, 4):
            out = np.eye(4, dtype=np.float64)
            out[:3, :] = H
            return out
        if H.shape == (3, 3):
            out = np.eye(4, dtype=np.float64)
            out[:3, :3] = H
            return out
        raise ValueError(f"Expected (4,4), (3,4) or (3,3), got {H.shape}")

    @staticmethod
    def _inv(T: np.ndarray) -> np.ndarray:
        return np.linalg.inv(T)

    @staticmethod
    def _parse_view(v: str):
        # "rgb_left" -> ("rgb","left")
        if "_" not in v:
            return None, None
        mod, side = v.rsplit("_", 1)
        return mod.lower(), side.lower()

    def _ms2_T_refL_to_modL(self, cross: Dict[str, Any], mod: str) -> Optional[np.ndarray]:
        ref = self.ms2_reference_mod
        mod = mod.lower()
        if mod == ref:
            return np.eye(4, dtype=np.float64)

        k_fwd = f"{ref.upper()}2{mod.upper()}"
        k_bwd = f"{mod.upper()}2{ref.upper()}"

        if k_fwd in cross and cross[k_fwd] is not None:
            return self._as_4x4(cross[k_fwd])
        if k_bwd in cross and cross[k_bwd] is not None:
            return self._inv(self._as_4x4(cross[k_bwd]))

        return None

    def _ms2_T_modL_to_modR(self, stereo: Dict[str, Any], mod: str) -> Optional[np.ndarray]:
        mod = mod.lower()
        if mod not in stereo:
            return None
        info = stereo[mod]
        if not isinstance(info, dict) or "extrinsics_L2R" not in info:
            return None
        return self._as_4x4(info["extrinsics_L2R"])

    def _build_from_cam2lidar(self, results: dict, view_names: list) -> Dict[str, Dict[str, np.ndarray]]:
        n = len(view_names)

        cam2lidar = results.get("cam2lidar", None)
        lidar2cam = results.get("lidar2cam", None)

        if cam2lidar is None and lidar2cam is None:
            raise KeyError("needs results['cam2lidar'] or results['lidar2cam']")

        cam2lidar_4 = [None] * n
        lidar2cam_4 = [None] * n

        if cam2lidar is not None:
            if isinstance(cam2lidar, np.ndarray):
                for i in range(n):
                    cam2lidar_4[i] = self._as_4x4(cam2lidar[i])
            else:
                for i in range(n):
                    cam2lidar_4[i] = None if cam2lidar[i] is None else self._as_4x4(cam2lidar[i])

            for i in range(n):
                if cam2lidar_4[i] is None:
                    continue
                lidar2cam_4[i] = self._inv(cam2lidar_4[i])

        else:
            if isinstance(lidar2cam, np.ndarray):
                for i in range(n):
                    lidar2cam_4[i] = self._as_4x4(lidar2cam[i])
            else:
                for i in range(n):
                    lidar2cam_4[i] = None if lidar2cam[i] is None else self._as_4x4(lidar2cam[i])

            for i in range(n):
                if lidar2cam_4[i] is None:
                    continue
                cam2lidar_4[i] = self._inv(lidar2cam_4[i])

        extrinsics: Dict[str, Dict[str, np.ndarray]] = {}
        for i, src in enumerate(view_names):
            extrinsics[src] = {}
            for j, dst in enumerate(view_names):
                if (not self.include_identity) and (i == j):
                    continue
                if cam2lidar_4[i] is None or lidar2cam_4[j] is None:
                    continue
                T_src_to_dst = (lidar2cam_4[j] @ cam2lidar_4[i]).astype(self.dtype, copy=False)
                extrinsics[src][dst] = T_src_to_dst
        return extrinsics

    def _build_ms2_fallback(self, results: dict, view_names: list) -> Dict[str, Dict[str, np.ndarray]]:
        cross = results.get("extrinsics", None)
        stereo = results.get("stereo", None)

        if not isinstance(cross, dict) or not isinstance(stereo, dict):
            raise KeyError("MS2 fallback needs results['extrinsics'] (flat dict) and results['stereo'].")

        # Build T_refLeft->view for each view
        T_ref_to_view: Dict[str, np.ndarray] = {}
        for v in view_names:
            mod, side = self._parse_view(v)
            if mod is None:
                continue

            T_refL_to_modL = self._ms2_T_refL_to_modL(cross, mod)
            if T_refL_to_modL is None:
                continue

            if side == "left":
                T_ref_to_view[v] = T_refL_to_modL
            elif side == "right":
                T_modL_to_modR = self._ms2_T_modL_to_modR(stereo, mod)
                if T_modL_to_modR is None:
                    continue
                T_ref_to_view[v] = (T_modL_to_modR @ T_refL_to_modL)

        extrinsics: Dict[str, Dict[str, np.ndarray]] = {}
        for src in view_names:
            if src not in T_ref_to_view:
                continue
            extrinsics[src] = {}
            T_src_to_ref = self._inv(T_ref_to_view[src])

            for dst in view_names:
                if (not self.include_identity) and (src == dst):
                    continue
                if dst not in T_ref_to_view:
                    continue
                T_ref_to_dst = T_ref_to_view[dst]
                extrinsics[src][dst] = (T_ref_to_dst @ T_src_to_ref).astype(self.dtype, copy=False)

        return extrinsics

    def _maybe_set_baselines(self, results: dict) -> None:
        """
        Ensures results has per-modality baseline keys like:
            results["rgb_baseline"], results["nir_baseline"], results["thr_baseline"], ...

        Priority:
          1) If already present -> keep.
          2) If results["stereo"][mod]["baseline"] exists -> use it.
          3) Otherwise do nothing.
        """
        stereo = results.get("stereo", None)
        if not isinstance(stereo, dict):
            return

        for mod, info in stereo.items():
            if not isinstance(info, dict):
                continue
            k = f"{mod}_baseline"
            if k in results and results[k] is not None:
                continue
            baseline = info.get("baseline", None)
            if baseline is None:
                continue
            results[k] = [float(baseline)]

    def transform(self, results: dict) -> Optional[dict]:
        if "view_names" not in results:
            return results

        view_names = results["view_names"]

        has_lidar_style = (results.get("cam2lidar", None) is not None) or (results.get("lidar2cam", None) is not None)

        if has_lidar_style:
            extr = self._build_from_cam2lidar(results, view_names)
        else:
            extr = self._build_ms2_fallback(results, view_names)

        self._maybe_set_baselines(results)

        results[self.store_key] = extr
        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(store_key='{self.store_key}', "
            f"dtype='{np.dtype(self.dtype).name}', include_identity={self.include_identity}, "
            f"ms2_reference_mod='{self.ms2_reference_mod}')"
        )

class CalculateTimestampDifferences(BaseTransform):
    """
    Build all camera-to-camera timestamp differences and store them into
    results[store_key] as a nested dict:

        results[store_key][src_view][dst_view] = ts_src - ts_dst

    Timestamps are extracted from each filename (aligned by index with view_names).
    Default extraction logic:
      1) take basename
      2) try: last '_' chunk before extension, must be digits
      3) fallback: last run of digits in the basename

    Optionally also stores into results['metas'][store_key] if available.
    """

    def __init__(
        self,
        store_key: str = "timestamp_diffs",
        filename_key: str = "filename",
        view_names_key: str = "view_names",
        dtype: str = "int64",
        include_identity: bool = True,
        also_store_in_metas: bool = True,
        metas_key: str = "metas",
    ):
        super().__init__()
        self.store_key = store_key
        self.filename_key = filename_key
        self.view_names_key = view_names_key
        self.dtype = np.int64 if dtype == "int64" else np.float64
        self.include_identity = include_identity
        self.also_store_in_metas = also_store_in_metas
        self.metas_key = metas_key

    @staticmethod
    def _extract_timestamp_from_filename(fname: Any) -> int:
        if fname is None:
            raise ValueError("Filename is None; cannot extract timestamp.")
        s = str(fname)
        base = os.path.basename(s)

        # Preferred: ..._<digits>.<ext>
        stem, _ext = os.path.splitext(base)
        last_chunk = stem.split("_")[-1]
        if last_chunk.isdigit():
            return int(last_chunk)

        # Fallback: last run of digits anywhere in basename
        m = re.findall(r"(\d+)", stem)
        if m:
            return int(m[-1])

        raise ValueError(f"Could not extract timestamp from filename: {fname!r}")

    def transform(self, results: dict) -> Optional[dict]:
        if self.view_names_key not in results:
            return results  # nothing to do

        view_names = results[self.view_names_key]
        n = len(view_names)

        if self.filename_key not in results:
            raise KeyError(
                f"{self.__class__.__name__} needs results['{self.filename_key}'] "
                f"aligned with results['{self.view_names_key}']."
            )

        filenames = results[self.filename_key]
        if isinstance(filenames, (str, bytes)):
            # single filename -> only valid if single view
            if n != 1:
                raise ValueError(
                    f"Got a single filename but {n} view_names. "
                    f"Expected a list of {n} filenames."
                )
            filenames = [filenames]

        if len(filenames) != n:
            raise ValueError(
                f"Length mismatch: {len(filenames)} filenames vs {n} view_names."
            )

        # Extract per-view timestamps
        timestamps = [None] * n
        for i in range(n):
            timestamps[i] = self._extract_timestamp_from_filename(filenames[i])

        # Build nested dict of differences
        diffs: Dict[str, Dict[str, Any]] = {}
        for i, src in enumerate(view_names):
            diffs[src] = {}
            for j, dst in enumerate(view_names):
                if (not self.include_identity) and (i == j):
                    continue
                diffs[src][dst] = (self.dtype(timestamps[i] - timestamps[j]))

        results[self.store_key] = diffs

        # Optionally also store into metas
        if self.also_store_in_metas and self.metas_key in results and isinstance(results[self.metas_key], dict):
            results[self.metas_key][self.store_key] = diffs

        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(store_key='{self.store_key}', "
            f"filename_key='{self.filename_key}', view_names_key='{self.view_names_key}', "
            f"dtype='{np.dtype(self.dtype).name}', include_identity={self.include_identity}, "
            f"also_store_in_metas={self.also_store_in_metas}, metas_key='{self.metas_key}')"
        )

def to_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)

class Pack3DDetInputs(BaseTransform):
    INPUTS_KEYS = ['points', 'img', 'img_gated', 'img_rgb', 'img_rccb']
    INSTANCEDATA_3D_KEYS = [
        'gt_bboxes_3d', 'gt_labels_3d', 'attr_labels', 'depths', 'centers_2d'
    ]
    INSTANCEDATA_2D_KEYS = [
        'gt_bboxes',
        'gt_bboxes_labels',
    ]

    SEG_KEYS = [
        'gt_seg_map', 'pts_instance_mask', 'pts_semantic_mask',
        'gt_semantic_seg'
    ]

    def __init__(
        self,
        keys: tuple,
        meta_keys: tuple = ('img_path', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'num_pts_feats', 'pcd_trans',
                            'sample_idx', 'pcd_scale_factor', 'pcd_rotation',
                            'pcd_rotation_angle', 'lidar_path',
                            'transformation_3d_flow', 'trans_mat',
                            'affine_aug', 'sweep_img_metas', 'ori_cam2img',
                            'cam2global', 'crop_offset', 'img_crop_offset',
                            'resize_img_shape', 'lidar2cam', 'ori_lidar2img',
                            'num_ref_frames', 'num_views', 'ego2global',
                            'axis_align_matrix')
    ) -> None:
        self.keys = keys
        self.meta_keys = meta_keys

    def _remove_prefix(self, key: str) -> str:
        if key.startswith('gt_'):
            key = key[3:]
        return key

    def transform(self, results: Union[dict,
                                       List[dict]]) -> Union[dict, List[dict]]:
        """Method to pack the input data. when the value in this dict is a
        list, it usually is in Augmentations Testing.

        Args:
            results (dict | list[dict]): Result dict from the data pipeline.

        Returns:
            dict | List[dict]:

            - 'inputs' (dict): The forward data of models. It usually contains
              following keys:

                - points
                - img

            - 'data_samples' (:obj:`Det3DDataSample`): The annotation info of
              the sample.
        """
        # augtest
        if isinstance(results, list):
            if len(results) == 1:
                # simple test
                return self.pack_single_results(results[0])
            pack_results = []
            for single_result in results:
                pack_results.append(self.pack_single_results(single_result))
            return pack_results
        # norm training and simple testing
        elif isinstance(results, dict):
            return self.pack_single_results(results)
        else:
            raise NotImplementedError

    def _stack_imgs_to_tensor(self, img_list):
        # your original logic, wrapped in a helper
        imgs = np.stack(img_list, axis=0)
        if imgs.flags.c_contiguous:
            imgs = to_tensor(imgs).permute(0, 3, 1, 2).contiguous()
        else:
            imgs = to_tensor(
                np.ascontiguousarray(imgs.transpose(0, 3, 1, 2))
            )
        return imgs

    def pack_single_results(self, results: dict) -> dict:
        """Method to pack the single input data. when the value in this dict is
        a list, it usually is in Augmentations Testing.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: A dict contains

            - 'inputs' (dict): The forward data of models. It usually contains
              following keys:

                - points
                - img

            - 'data_samples' (:obj:`Det3DDataSample`): The annotation info
              of the sample.
        """
        # Format 3D data
        if 'points' in results:
            if isinstance(results['points'], BasePoints):
                results['points'] = results['points'].tensor

        if 'img' in results:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame

                view_names = results.get('view_names', None)

                if not view_names:
                    # fallback: old behavior if we don't have view_names
                    results['img'] = self._stack_imgs_to_tensor(results['img'])
                else:
                    assert len(results['img']) == len(view_names), \
                        "len(img) and len(view_names) must match"

                    # group images by cam_type (prefix before first "_")
                    grouped_imgs = defaultdict(list)
                    for img, name in zip(results['img'], view_names):
                        cam_type = name.split("_")[0]
                        grouped_imgs[cam_type].append(img)

                    # create stacked outputs per cam_type
                    for cam_type, img_list in grouped_imgs.items():
                        key = f'img_{cam_type}'  # e.g. img_gated, img_rccb, img_rgb
                        results[key] = self._stack_imgs_to_tensor(img_list)
            else:
                img = results['img']
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                # To improve the computational speed by by 3-5 times, apply:
                # `torch.permute()` rather than `np.transpose()`.
                # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
                # for more details
                if img.flags.c_contiguous:
                    img = to_tensor(img).permute(2, 0, 1).contiguous()
                else:
                    img = to_tensor(
                        np.ascontiguousarray(img.transpose(2, 0, 1)))
                results['img'] = img

        for key in [
                'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                'gt_bboxes_labels', 'attr_labels', 'pts_instance_mask',
                'pts_semantic_mask', 'centers_2d', 'depths', 'gt_labels_3d'
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = [to_tensor(res) for res in results[key]]
            else:
                results[key] = to_tensor(results[key])
        if 'gt_bboxes_3d' in results:
            if not isinstance(results['gt_bboxes_3d'], BaseInstance3DBoxes):
                results['gt_bboxes_3d'] = to_tensor(results['gt_bboxes_3d'])

        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = to_tensor(
                results['gt_semantic_seg'][None])
        if 'gt_seg_map' in results:
            results['gt_seg_map'] = results['gt_seg_map'][None, ...]

        data_sample = {} #Det3DDataSample()
        gt_instances_3d = InstanceData()
        gt_instances = InstanceData()
        gt_pts_seg = PointData()

        data_metas = {}
        for key in self.meta_keys:
            if key in results:
                data_metas[key] = results[key]
            elif 'images' in results:
                if len(results['images'].keys()) == 1:
                    cam_type = list(results['images'].keys())[0]
                    # single-view image
                    if key in results['images'][cam_type]:
                        data_metas[key] = results['images'][cam_type][key]
                else:
                    # multi-view image
                    img_metas = []
                    cam_types = list(results['images'].keys())
                    for cam_type in cam_types:
                        if key in results['images'][cam_type]:
                            img_metas.append(results['images'][cam_type][key])
                    if len(img_metas) > 0:
                        data_metas[key] = img_metas
            elif 'lidar_points' in results:
                if key in results['lidar_points']:
                    data_metas[key] = results['lidar_points'][key]
        data_sample = data_metas

        inputs = {}
        for key in self.keys:
            if key in results:
                if key in self.INPUTS_KEYS:
                    inputs[key] = results[key]
                elif key in self.INSTANCEDATA_3D_KEYS:
                    gt_instances_3d[self._remove_prefix(key)] = results[key]
                elif key in self.INSTANCEDATA_2D_KEYS:
                    if key == 'gt_bboxes_labels':
                        gt_instances['labels'] = results[key]
                    else:
                        gt_instances[self._remove_prefix(key)] = results[key]
                elif key in self.SEG_KEYS:
                    gt_pts_seg[self._remove_prefix(key)] = results[key]
                else:
                    raise NotImplementedError(f'Please modified '
                                              f'`Pack3DDetInputs` '
                                              f'to put {key} to '
                                              f'corresponding field')


        packed_results = dict()
        packed_results['data_samples'] = data_sample
        packed_results['inputs'] = inputs

        return packed_results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys})'
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


class LongRangeBEVPack3DDetInputs(Pack3DDetInputs):
    INPUTS_KEYS = [
        # generic
        'points', 'img', 'gt_points',
        'sparse_depth', 'gt_sparse_depth',
        'gt_semantic_occ', 'depth',

        # images (per modality)
        'img_gated', 'img_rgb', 'img_rccb',

        # sparse depth (per modality)
        'sparse_depth_gated', 'sparse_depth_rgb', 'sparse_depth_rccb',

        # gt sparse depth (per modality)
        'gt_sparse_depth_gated', 'gt_sparse_depth_rgb', 'gt_sparse_depth_rccb',

        # HD sparse depth (per modality)
        'sparse_depth_gated_hd', 'sparse_depth_rgb_hd', 'sparse_depth_rccb_hd',

        # HD gt sparse depth (per modality)
        'gt_sparse_depth_gated_hd', 'gt_sparse_depth_rgb_hd', 'gt_sparse_depth_rccb_hd',
    ]

    INSTANCEDATA_2D_KEYS = [
        'gt_bboxes', 'gt_bboxes_labels', 'calculated_truncation',
        'occlusion', 'score_2d'
    ]

    def transform(self, *args):
        # keep behavior identical to parent
        return super().transform(*args)

    def _stack_to_tensor(self, arr_list):
        arr = np.stack(arr_list, axis=0)
        if arr.flags.c_contiguous:
            arr = to_tensor(arr).permute(0, 3, 1, 2).contiguous()
        else:
            arr = to_tensor(
                np.ascontiguousarray(arr.transpose(0, 3, 1, 2))
            )
        return arr

    def pack_single_results(self, results: dict) -> dict:
        # optional extra tensors specific to your pipeline:

        if 'gt_semantic_occ' in results:
            results['gt_semantic_occ'] = to_tensor(results['gt_semantic_occ'])

        if 'gt_points' in results and hasattr(results['gt_points'], 'tensor'):
            results['gt_points'] = results['gt_points'].tensor

        # Handle sparse_depth (accept list of HxWxC or a single HxWxC/ HxW)
        if 'sparse_depth' in results:
            if isinstance(results['sparse_depth'], list):

                view_names = results.get('view_names', None)

                if not view_names:
                    # fallback: old behavior
                    results['sparse_depth'] = self._stack_to_tensor(results['sparse_depth'])
                else:
                    assert len(results['sparse_depth']) == len(view_names), \
                        "len(sparse_depth) and len(view_names) must match"

                    # group by camera prefix (e.g. gated, rccb, rgb…)
                    grouped = defaultdict(list)
                    for depth, name in zip(results['sparse_depth'], view_names):
                        cam_type = name.split("_")[0]
                        grouped[cam_type].append(depth)

                    for cam_type, arr_list in grouped.items():
                        key = f'sparse_depth_{cam_type}'
                        results[key] = self._stack_to_tensor(arr_list)
            else:
                img = results['sparse_depth']
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                if img.flags.c_contiguous:
                    img = to_tensor(img).permute(2, 0, 1).contiguous()
                else:
                    img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
                results['sparse_depth'] = img

        if 'gt_sparse_depth' in results:
            if isinstance(results['gt_sparse_depth'], list):

                view_names = results.get('view_names', None)

                if not view_names:
                    # fallback: old behavior
                    results['gt_sparse_depth'] = self._stack_to_tensor(results['gt_sparse_depth'])
                else:
                    assert len(results['gt_sparse_depth']) == len(view_names), \
                        "len(gt_sparse_depth) and len(view_names) must match"

                    # group by cam_type
                    grouped = defaultdict(list)
                    for depth, name in zip(results['gt_sparse_depth'], view_names):
                        cam_type = name.split("_")[0]  # gated, rccb, rgb, ...
                        grouped[cam_type].append(depth)

                    for cam_type, arr_list in grouped.items():
                        key = f'gt_sparse_depth_{cam_type}'
                        results[key] = self._stack_to_tensor(arr_list)
            else:
                img = results['gt_sparse_depth']
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                if img.flags.c_contiguous:
                    img = to_tensor(img).permute(2, 0, 1).contiguous()
                else:
                    img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
                results['gt_sparse_depth'] = img

        # Handle sparse_depth (accept list of HxWxC or a single HxWxC/ HxW)
        if 'sparse_depth_hd' in results:
            if isinstance(results['sparse_depth_hd'], list):

                view_names = results.get('view_names', None)

                if not view_names:
                    # fallback: old behavior
                    results['sparse_depth_hd'] = self._stack_to_tensor(results['sparse_depth_hd'])
                else:
                    assert len(results['sparse_depth_hd']) == len(view_names), \
                        "len(sparse_depth) and len(view_names) must match"

                    # group by camera prefix (e.g. gated, rccb, rgb…)
                    grouped = defaultdict(list)
                    for depth, name in zip(results['sparse_depth_hd'], view_names):
                        cam_type = name.split("_")[0]
                        grouped[cam_type].append(depth)

                    for cam_type, arr_list in grouped.items():
                        key = f'sparse_depth_{cam_type}_hd'
                        results[key] = self._stack_to_tensor(arr_list)
            else:
                img = results['sparse_depth_hd']
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                if img.flags.c_contiguous:
                    img = to_tensor(img).permute(2, 0, 1).contiguous()
                else:
                    img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
                results['sparse_depth'] = img

        if 'gt_sparse_depth_hd' in results:
            if isinstance(results['gt_sparse_depth_hd'], list):

                view_names = results.get('view_names', None)

                if not view_names:
                    # fallback: old behavior
                    results['gt_sparse_depth_hd'] = self._stack_to_tensor(results['gt_sparse_depth_hd'])
                else:
                    assert len(results['gt_sparse_depth_hd']) == len(view_names), \
                        "len(gt_sparse_depth_hd) and len(view_names) must match"

                    # group by cam_type
                    grouped = defaultdict(list)
                    for depth, name in zip(results['gt_sparse_depth_hd'], view_names):
                        cam_type = name.split("_")[0]  # gated, rccb, rgb, ...
                        grouped[cam_type].append(depth)

                    for cam_type, arr_list in grouped.items():
                        key = f'gt_sparse_depth_{cam_type}_hd'
                        results[key] = self._stack_to_tensor(arr_list)
            else:
                img = results['gt_sparse_depth_hd']
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                if img.flags.c_contiguous:
                    img = to_tensor(img).permute(2, 0, 1).contiguous()
                else:
                    img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
                results['gt_sparse_depth_hd'] = img

        if 'depth' in results:
            if isinstance(results['depth']['mean'], list):
                imgs = np.stack(results['depth']['mean'], axis=0)
                if imgs.flags.c_contiguous:
                    imgs = to_tensor(imgs).permute(0, 3, 1, 2).contiguous()
                else:
                    imgs = to_tensor(np.ascontiguousarray(imgs.transpose(0, 3, 1, 2)))
                results['depth']['mean'] = imgs
            else:
                img = results['depth']['mean']
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                if img.flags.c_contiguous:
                    img = to_tensor(img).permute(2, 0, 1).contiguous()
                else:
                    img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
                results['depth']['mean'] = img
            if isinstance(results['depth']['std'], list):
                imgs = np.stack(results['depth']['std'], axis=0)
                if imgs.flags.c_contiguous:
                    imgs = to_tensor(imgs).permute(0, 3, 1, 2).contiguous()
                else:
                    imgs = to_tensor(np.ascontiguousarray(imgs.transpose(0, 3, 1, 2)))
                results['depth']['std'] = imgs
            else:
                img = results['depth']['std']
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                if img.flags.c_contiguous:
                    img = to_tensor(img).permute(2, 0, 1).contiguous()
                else:
                    img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
                results['depth']['std'] = img

        # Delegate to parent to finish packing
        return super().pack_single_results(results)
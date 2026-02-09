import os
import copy
import pickle
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.data_utils import load, _build_pipeline, get_dist_info

BatchScene = Tuple[Union[str, int], Union[str, int]]


class GatedRCCBStereoDataset(Dataset):
    """Bare-bones depth dataset with multi-frame support.

    Differences to TorcDataset:
      - No MMDet/MMEngine-style meta / data-sample classes.
      - No 2D/3D boxes, labels, or category stats.
      - Only cares about depth-ish things: lidar, projected lidar, images,
        intrinsics, extrinsics, etc.

    Still supports:
      * queue_length (past frames)
      * future_length (future frames)
      * transform pipeline
      * __getitem__ -> {"inputs": ..., "data_samples": ...}

    You can control which keys are kept via
        * return_input_keys
        * return_data_sample_keys
    so it's easy to extend without custom DataSample classes.
    """

    def __init__(
        self,
        data_root: Optional[Union[str, Path]],
        ann_file: str,
        *,
        ann_file_val: str = None,
        data_prefix: Optional[Dict[str, str]] = None,
        pipeline: Optional[List[Union[dict, Callable]]] = None,
        modality: Optional[Dict[str, bool]] = None,
        default_cam_key: Optional[str] = "forward_center_medium_ID065",
        test_mode: bool = False,
        # temporal sampling
        queue_length: int = 0,
        future_length: int = 0,
        load_frame_interval: Optional[int] = None,
        rand_frame_interval: Sequence[int] = (1,),
        deterministic_frame_interval: int = 1,
        deterministic_frame_interval_future: Optional[int] = None,
        use_random_interval: bool = False,
        # dataset selection / slicing
        batch_scene_pattern: Optional[Union[BatchScene, Sequence[BatchScene]]] = None,
        indices: Optional[Union[int, Sequence[int]]] = None,
        filter_cfg: Optional[dict] = None,  # reserved hook
        # misc
        batch_size: int = 1,
        max_refetch: int = 1000,
        skip_n_first_and_last_samples: bool = False,
        overfit_single_image: bool = False,
        # return-shape control
        return_input_keys: Optional[Sequence[str]] = None,
        return_data_sample_keys: Optional[Sequence[str]] = None,
        point_keys: Sequence[str] = ("points",),  # which keys are variable-length point clouds
        # caching of parsed annotation file
        load_cached_data_list: bool = False,
        save_cached_data_list: bool = False,
        data_list_path: str = "",
        num_samples: int = -1,
    ) -> None:
        super().__init__()

        self.data_root = os.fspath(data_root) if data_root is not None else None
        self.ann_file = ann_file if ((not test_mode) or (ann_file_val is None)) else ann_file_val
        self.data_prefix: Dict[str, str] = dict(data_prefix or {})  # shallow copy
        self.pipeline_cfg = pipeline or []

        # modality: {use_lidar: bool, use_camera: bool}
        self.modality = modality or {}
        if "use_lidar" not in self.modality:
            self.modality["use_lidar"] = True
        if "use_camera" not in self.modality:
            self.modality["use_camera"] = True
        assert (
            self.modality["use_lidar"] or self.modality["use_camera"]
        ), "At least one of use_lidar / use_camera must be True."

        self.default_cam_key = default_cam_key
        self.test_mode = test_mode

        # temporal sampling
        self.queue_length = queue_length
        self.future_length = future_length
        self.load_frame_interval = load_frame_interval
        self.rand_frame_interval = tuple(rand_frame_interval)
        self.deterministic_frame_interval = deterministic_frame_interval
        self.deterministic_frame_interval_future = (
            deterministic_frame_interval_future
            if deterministic_frame_interval_future is not None
            else deterministic_frame_interval
        )
        self.use_random_interval = use_random_interval
        self.no_future = True if future_length == 0 else False

        # selection / slicing
        self.batch_scene_patterns = batch_scene_pattern
        self._indices = indices
        self.filter_cfg = filter_cfg

        # misc
        self.batch_size = batch_size
        self.max_refetch = max_refetch
        self.skip_n_first_and_last_samples = skip_n_first_and_last_samples
        self.overfit_single_image = overfit_single_image

        # return-key control
        self.return_input_keys = list(return_input_keys) if return_input_keys is not None else None
        self.return_data_sample_keys = (
            list(return_data_sample_keys) if return_data_sample_keys is not None else None
        )
        self.point_keys = tuple(point_keys)

        # annotation caching
        self.load_cached_data_list = load_cached_data_list
        self.save_cached_data_list = save_cached_data_list
        self.data_list_path = data_list_path
        self.num_samples = num_samples

        # scene caches (for skip_n_first_and_last_samples heuristic)
        self._scene_cache: Dict[Tuple[Any, Any], List[dict]] = {}
        self._scene_index_cache: Dict[Tuple[Any, Any], Dict[int, int]] = {}

        # data
        self.data_list: List[dict] = []
        self.usable_index: List[int] = []

        # join prefixes
        self._join_prefix()

        # build transform pipeline
        self.pipeline = _build_pipeline(self.pipeline_cfg)

        # metainfo from ann_file (optional, not used for labels here)
        self.metainfo: Dict[str, Any] = {}

        # full initialization
        self._fully_initialized = False
        self.full_init()

        # build usable indices given queue_length / future_length
        self._build_temporal_index()

    # ==================================================================
    # basic API
    # ==================================================================
    def __len__(self) -> int:
        return len(self.usable_index)

    # ==================================================================
    # batch / scene pattern helpers (subset of your original)
    # ==================================================================
    @staticmethod
    def _to_pairs(
        bsp: Optional[Union[BatchScene, Sequence[BatchScene]]]
    ) -> Sequence[BatchScene]:
        if bsp is None:
            return []
        if isinstance(bsp, tuple):
            return [bsp]
        if isinstance(bsp, Sequence) and not isinstance(bsp, (str, bytes)):
            return list(bsp)
        raise TypeError("batch_scene_pattern must be (batch, scene) or a sequence of such tuples")

    @staticmethod
    def _augment_token(s: Union[str, int]) -> Iterable[str]:
        toks = re.split(r"[-_.]", str(s).strip())
        for sep in (".", "-", "_"):
            yield sep.join(toks)


    def should_keep(self, raw_data_info: dict) -> bool:
        """Return True if this sample should be kept given batch_scene_pattern."""
        if not self.batch_scene_patterns:
            return True
        recording = str(raw_data_info.get("recording"))
        index = raw_data_info.get("index")
        if list(self.batch_scene_patterns)[0][1] in [None, 'None']:
            # match on batch only
            return any(recording == b for b, _ in self.batch_scene_patterns)
        key = (recording, str(index))
        return key in self.batch_scene_patterns

    # ==================================================================
    # path joining and ann loading
    # ==================================================================
    def _join_prefix(self) -> None:
        """Join data_root with data_prefix and ann_file."""
        if self.data_root is None:
            return

        def _is_abs(p: str) -> bool:
            return os.path.isabs(p)

        def _join(p: str) -> str:
            return os.path.normpath(os.path.join(self.data_root, p))

        # ann_file
        if self.ann_file and not _is_abs(self.ann_file):
            self.ann_file = _join(self.ann_file)

        # data_prefix per key
        for k, v in list(self.data_prefix.items()):
            if not isinstance(v, str):
                raise TypeError(f"data_prefix[{k!r}] must be str, got {type(v)}")
            if not _is_abs(v):
                self.data_prefix[k] = _join(v)
            else:
                self.data_prefix[k] = os.path.normpath(v)

    def full_init(self) -> None:
        if self._fully_initialized:
            return

        if self.load_cached_data_list and self.data_list_path:
            cached = load(self.data_list_path)
            self.data_list = cached["data_list"]
            self.metainfo = cached.get("metainfo", {})
        else:
            self.data_list = self.load_data_list_from_ann()
            if self.save_cached_data_list and self.data_list_path:
                os.makedirs(os.path.dirname(self.data_list_path), exist_ok=True)
                with open(self.data_list_path, "wb") as f:
                    pickle.dump(
                        {"data_list": self.data_list, "metainfo": self.metainfo},
                        f,
                    )

        # external slicing
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)

        self._fully_initialized = True

    def load_data_list_from_ann(self) -> List[dict]:
        annotations = load(self.ann_file)
        if not isinstance(annotations, dict):
            raise TypeError(f"Annotation file should contain a dict, got {type(annotations)}")

        raw_data_list: List[dict] = annotations.pop("samples")
        self.metainfo = annotations

        if self.num_samples != -1:
            raw_data_list = raw_data_list[:self.num_samples]

        parsed_list: List[dict] = []

        print("Parsing raw data_list ...")
        for raw_data_info in tqdm(raw_data_list):
            if not self.should_keep(raw_data_info):
                continue

            parsed = self.parse_data_info(raw_data_info)
            if isinstance(parsed, dict):
                parsed_list.append(parsed)
            elif isinstance(parsed, list):
                for item in parsed:
                    if not isinstance(item, dict):
                        raise TypeError(
                            f"parse_data_info must return dict or list[dict], got list[{type(item)}]"
                        )
                    parsed_list.append(item)
            else:
                raise TypeError(f"parse_data_info must return dict or list[dict], got {type(parsed)}")

        return parsed_list

    def _ensure_scene_cache(self, raw_data_list: List[dict], key: Tuple[Any, Any]) -> None:
        if key in self._scene_cache:
            return
        batch, scene = key
        scenes = [x for x in raw_data_list if x.get("batch_id") == batch and x.get("scene_id") == scene]
        scenes.sort(key=lambda x: x["timestamp"])
        self._scene_cache[key] = scenes
        self._scene_index_cache[key] = {id(x): i for i, x in enumerate(scenes)}

    def _get_unserialized_subset(self, indices: Union[Sequence[int], int]) -> List[dict]:
        if isinstance(indices, int):
            if indices >= 0:
                return self.data_list[: indices]
            else:
                return self.data_list[indices:]
        if isinstance(indices, Sequence):
            return [self.data_list[i] for i in indices]
        raise TypeError(f"indices should be int or Sequence[int], got {type(indices)}")

    # ==================================================================
    # data_info parsing: join paths & expose intrinsics/extrinsics
    # ==================================================================
    def _selective_copy_info(self, info: dict) -> dict:
        out = info.copy()
        if isinstance(out.get("lidar_points"), dict):
            out["lidar_points"] = out["lidar_points"].copy()
        if isinstance(out.get("lidar_sweeps"), list):
            sweeps = []
            for s in out["lidar_sweeps"]:
                s2 = s.copy()
                if isinstance(s2.get("lidar_points"), dict):
                    s2["lidar_points"] = s2["lidar_points"].copy()
                sweeps.append(s2)
            out["lidar_sweeps"] = sweeps
        if isinstance(out.get("images"), dict):
            out["images"] = {
                cam: (img.copy() if isinstance(img, dict) else img)
                for cam, img in out["images"].items()
            }
        return out

    def _parse_lidar_info(self, info: dict) -> None:
        lidar_info = info["lidar_points"]
        lidar_path = lidar_info["lidar_path"]

        # NOTE: this is intentionally simple; adapt to your prefix layout if needed.
        if self.data_root is not None and not os.path.isabs(lidar_path):
            lidar_path = os.path.join(self.data_root, lidar_path)

        lidar_info["lidar_path"] = lidar_path
        info["lidar_path"] = lidar_path
        info["num_pts_feats"] = lidar_info.get("num_pts_feats", 4)

    def get_corrected_absolute_path(self, original_path: Union[str, Path], cam_id: str) -> str:
        """Return an absolute image path under data_root.

        Currently simplified: we just join with data_root.
        """
        original_path = os.fspath(original_path)
        if self.data_root is None or os.path.isabs(original_path):
            return original_path
        return os.path.join(self.data_root, original_path)

    def _parse_camera_info(self, info: dict) -> None:
        if "img_filename" not in info:
            info["img_filename"] = []

        if "images" not in info:
            return

        for cam_id, img_info in info["images"].items():
            if "img_path" in img_info and img_info["img_path"] is not None:
                img_info["img_path"] = self.get_corrected_absolute_path(img_info["img_path"], cam_id)
            else:
                img_info["img_path"] = None

        if self.default_cam_key is not None:
            self._parse_default_camera_info(info)

    def _parse_default_camera_info(self, info: dict) -> None:
        if self.default_cam_key not in info.get("images", {}):
            raise ValueError(f"default_cam_key={self.default_cam_key!r} not in info['images']")
        cam_info = info["images"][self.default_cam_key]
        cam2img = cam_info.get("cam2img")
        if cam2img is None:
            raise ValueError(f"default_cam_key={self.default_cam_key!r} has no 'cam2img'")
        cam2img = np.array(cam2img)
        if cam2img.shape == (3, 3):
            cam2img = np.pad(cam2img, (0, 1))
            cam2img[-1, -1] = 1
        elif cam2img.shape != (4, 4):
            raise ValueError(
                f"default_cam_key={self.default_cam_key!r} invalid cam2img shape {cam2img.shape}"
            )
        cam_info["cam2img"] = cam2img

    def parse_data_info(self, info: dict) -> dict:
        """Legacy code, does nothing here.
        """
        return info

    # ==================================================================
    # temporal index building (queue_length / future_length)
    # ==================================================================
    def _build_temporal_index(self) -> None:
        """Build self.usable_index based on queue_length / future_length."""
        if self.queue_length == 0 and self.future_length == 0:
            usable_index = list(range(len(self.data_list)))
        else:
            last_scene_key = None
            last_scene_frame = -1
            usable_index: List[int] = []

            if self.use_random_interval and not self.test_mode:
                max_interval = max(self.rand_frame_interval)
                multiply_queues = max_interval
                multiply_queues_future = max_interval
            else:
                multiply_queues = self.deterministic_frame_interval
                multiply_queues_future = self.deterministic_frame_interval_future

            valid_prev = self.queue_length * multiply_queues
            valid_future = (self.future_length + 1) * multiply_queues_future

            # group per scene for consistent ordering
            scene_dict: Dict[Tuple[Any, Any], List[dict]] = {}
            for info in self.data_list:
                key = (info.get("batch_id"), info.get("scene_id"))
                scene_dict.setdefault(key, []).append(info)

            ordered_list: List[dict] = []
            for key, data_v in scene_dict.items():
                timestamps_list = [int(d["timestamp"]) for d in data_v]
                idxs = sorted(range(len(timestamps_list)), key=lambda k: timestamps_list[k])
                for i in idxs:
                    ordered_list.append(data_v[i])
            self.data_list = ordered_list

            if self.no_future:
                for index, info in enumerate(self.data_list):
                    scene_key = (info.get("batch_id"), info.get("scene_id"))
                    if scene_key != last_scene_key:
                        last_scene_key = scene_key
                        last_scene_frame = -1
                    last_scene_frame += 1
                    if last_scene_frame >= valid_prev:
                        usable_index.append(index)
            else:
                for index, info in enumerate(self.data_list):
                    scene_key = (info.get("batch_id"), info.get("scene_id"))
                    if scene_key != last_scene_key:
                        last_scene_key = scene_key
                        last_scene_frame = -1
                    last_scene_frame += 1
                    if last_scene_frame < valid_prev:
                        continue

                    tgt_future_index = index + valid_future
                    if tgt_future_index >= len(self.data_list):
                        break
                    future_info = self.data_list[tgt_future_index]
                    if (future_info.get("batch_id"), future_info.get("scene_id")) != scene_key:
                        continue
                    usable_index.append(index)

        if self.load_frame_interval is not None and self.load_frame_interval > 1:
            usable_index = usable_index[:: self.load_frame_interval]

        if self.overfit_single_image and len(usable_index) > 0:
            self.usable_index = usable_index[0:1] * 500
        else:
            self.usable_index = usable_index

        # pad to batch_size * world_size for DDP
        _, world_size = get_dist_info()
        batch_total = self.batch_size * world_size
        if batch_total > 0:
            modulo = len(self.usable_index) % batch_total
            if modulo != 0 and len(self.usable_index) > 0:
                pad = batch_total - modulo
                self.usable_index = self.usable_index + [self.usable_index[-1]] * pad

    # ==================================================================
    # public API: get_data_info / __getitem__
    # ==================================================================
    def get_data_info(self, index: int) -> dict:
        info = self.data_list[index]
        return info

    def _prepare_data_info(self, index: int) -> Optional[dict]:
        """Fetch a sample + its temporal neighbors and merge them."""
        if self.queue_length == 0 and self.future_length == 0:
            curr = self.prepare_data(index)
            if curr is None:
                return None
            return self._union2one(curr, [curr], [curr])

        if self.use_random_interval and not self.test_mode:
            interval = int(np.random.choice(self.rand_frame_interval, 1)[0])
            interval_future = interval
        else:
            interval = self.deterministic_frame_interval
            interval_future = self.deterministic_frame_interval_future

        # past
        previous_queue: List[dict] = []
        prev_indices = list(range(index - self.queue_length * interval, index, interval))
        prev_indices = sorted(prev_indices)
        for i in prev_indices:
            if i < 0:
                return None
            ex = self.prepare_data(i)
            if ex is None:
                return None
            previous_queue.append(ex)

        # future
        future_queue: List[dict] = []
        future_indices = list(range(index, index + (self.future_length + 1) * interval_future, interval_future))
        future_indices = sorted(future_indices)
        for i in future_indices:
            if i >= len(self.data_list):
                return None
            ex = self.prepare_data(i)
            if ex is None:
                return None
            future_queue.append(ex)

        if self.no_future:
            curr = self.prepare_data(index)
            if curr is None:
                return None
            return self._union2one(curr, previous_queue, [curr])

        if self.queue_length == 0:
            # only future frames
            return self._union2one(future_queue[0], [future_queue[0]], future_queue[1:])

        return self._union2one(future_queue[0], previous_queue, future_queue[1:])

    def __getitem__(self, idx: int) -> dict:
        if not self._fully_initialized:
            self.full_init()

        real_index = self.usable_index[idx]

        if self.test_mode:
            data = self._prepare_data_info(real_index)
            if data is None:
                raise RuntimeError("Test pipeline must not return None")
            return self._filter_return_keys(data)

        for _ in range(self.max_refetch + 1):
            data = self._prepare_data_info(real_index)
            if data is None:
                # broken sample or augmentation, resample
                real_index = self._rand_another()
                continue
            return self._filter_return_keys(data)

        raise RuntimeError(f"Cannot find valid sample after {self.max_refetch} retries")

    def _rand_another(self) -> int:
        return int(np.random.randint(0, len(self.usable_index)))

    def _filter_return_keys(self, data: dict) -> dict:
        """Apply return_input_keys / return_data_sample_keys filters."""
        inputs = data.get("inputs", {})
        data_samples = data.get("data_samples", {})

        if self.return_input_keys is not None:
            inputs = {k: v for k, v in inputs.items() if k in self.return_input_keys}
        if self.return_data_sample_keys is not None and isinstance(data_samples, dict):
            data_samples = {k: v for k, v in data_samples.items() if k in self.return_data_sample_keys}

        return {"inputs": inputs, "data_samples": data_samples}

    # ==================================================================
    # pipeline + merging
    # ==================================================================
    def prepare_data(self, index: int) -> Optional[dict]:
        """Call pipeline on the raw info dict at ``index``."""
        ori_input_dict = self.get_data_info(index)
        input_dict = copy.deepcopy(ori_input_dict)

        # e.g. input_dict["box_type_3d"] = "LiDAR" (if you still care)

        example = self.pipeline(input_dict)
        if example is None:
            return None

        # Contract: pipeline returns dict with:
        #   example["inputs"]       -> dict of model inputs
        #   example["data_samples"] -> dict of labels / meta (optional)
        if "inputs" not in example:
            raise KeyError("Pipeline must return a dict with an 'inputs' key")
        if "data_samples" not in example:
            example["data_samples"] = {}

        return example

    def _pad_and_stack_points(self, queue: List[dict], key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        tensors = [each["inputs"][key] for each in queue]
        lengths = [t.shape[0] for t in tensors]
        max_len = max(lengths)
        padded = []
        for t in tensors:
            pad = max_len - t.shape[0]
            if pad > 0:
                pad_tensor = torch.zeros((pad, t.shape[-1]), dtype=t.dtype, device=t.device)
                padded.append(torch.cat([t, pad_tensor], dim=0))
            else:
                padded.append(t)
        stacked = torch.stack(padded, dim=0)
        return stacked, torch.as_tensor(lengths, dtype=torch.long)

    def _union2one(
        self,
        curr: dict,
        previous_queue: List[dict],
        future_queue: List[dict],
    ) -> dict:
        """Merge curr / past / future samples into a single dict.

        Inputs:
            curr / previous_queue / future_queue come directly from the pipeline
            and must follow the contract in prepare_data().
        """
        inputs_out: Dict[str, Any] = {}
        data_samples_out: Dict[str, Any] = {}

        # ----------------- 1) inputs -----------------
        curr_inputs = curr.get("inputs", {})
        for key, value in curr_inputs.items():
            # special handling for variable-length point tensors
            if key in self.point_keys and torch.is_tensor(value):
                past_points, past_len = self._pad_and_stack_points(previous_queue, key)
                future_points, future_len = self._pad_and_stack_points(future_queue, key)
                inputs_out[key] = value
                inputs_out[f"past_{key}"] = past_points
                inputs_out[f"past_{key}_length"] = past_len
                inputs_out[f"future_{key}"] = future_points
                inputs_out[f"future_{key}_length"] = future_len
                continue

            # simple tensors (img, sparse_depth, projected_depth, ...)
            if torch.is_tensor(value):
                past_vals = [s["inputs"][key] for s in previous_queue if key in s["inputs"]]
                future_vals = [s["inputs"][key] for s in future_queue if key in s["inputs"]]
                past_vals = torch.stack(past_vals, dim=0) if past_vals else None
                future_vals = torch.stack(future_vals, dim=0) if future_vals else None
                inputs_out[key] = value
                inputs_out[f"past_{key}"] = past_vals
                inputs_out[f"future_{key}"] = future_vals
                continue

            # dict-of-tensors (e.g. depth["mean"] / depth["std"])
            if isinstance(value, dict):
                inputs_out[key] = value
                past_dict: Dict[str, Any] = {}
                future_dict: Dict[str, Any] = {}
                for sub_key, sub_val in value.items():
                    if torch.is_tensor(sub_val):
                        past_vals = [
                            s["inputs"][key][sub_key]
                            for s in previous_queue
                            if key in s["inputs"] and sub_key in s["inputs"][key]
                        ]
                        future_vals = [
                            s["inputs"][key][sub_key]
                            for s in future_queue
                            if key in s["inputs"] and sub_key in s["inputs"][key]
                        ]
                        if past_vals:
                            past_dict[sub_key] = torch.stack(past_vals, dim=0)
                        if future_vals:
                            future_dict[sub_key] = torch.stack(future_vals, dim=0)
                    else:
                        # non-tensor -> keep list
                        past_dict[sub_key] = [
                            s["inputs"][key][sub_key]
                            for s in previous_queue
                            if key in s["inputs"] and sub_key in s["inputs"][key]
                        ]
                        future_dict[sub_key] = [
                            s["inputs"][key][sub_key]
                            for s in future_queue
                            if key in s["inputs"] and sub_key in s["inputs"][key]
                        ]
                inputs_out[f"past_{key}"] = past_dict
                inputs_out[f"future_{key}"] = future_dict
                continue

            # fallback: keep current value + list of past/future values
            inputs_out[key] = value
            inputs_out[f"past_{key}"] = [
                s["inputs"][key]
                for s in previous_queue
                if key in s["inputs"]
            ]
            inputs_out[f"future_{key}"] = [
                s["inputs"][key]
                for s in future_queue
                if key in s["inputs"]
            ]

        # ----------------- 2) data_samples -----------------
        curr_samples = curr.get("data_samples", {})

        data_samples_out.update(curr_samples)

        return {"inputs": inputs_out, "data_samples": data_samples_out}
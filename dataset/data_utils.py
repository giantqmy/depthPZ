from __future__ import annotations
import io
import json
import pickle
from pathlib import Path
from typing import Any, Optional, Union, Tuple, List, ClassVar, Dict, Callable
from io import StringIO
from urllib.parse import urlparse
import warnings
import gzip
import bz2
import lzma
import urllib.request
from enum import Enum
from omegaconf import DictConfig, ListConfig, OmegaConf
import torch.distributed as dist
import torch
from hydra.utils import instantiate


try:
    import yaml  # PyYAML
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

_StrPath = Union[str, Path]

def load(
    file: Union[_StrPath, io.IOBase],
    file_format: Optional[str] = None,
    file_client_args: Optional[dict] = None,   # ignored (compat)
    backend_args: Optional[dict] = None,       # ignored (compat)
    **kwargs,
) -> Any:
    """
    Load data from json/yaml/pickle files, from a local path or a file-like object.

    Args:
        file: Path (str/Path) or an open file-like object supporting .read().
        file_format: 'json' | 'yaml'/'yml' | 'pickle'/'pkl'. If None, inferred from extension.
        file_client_args, backend_args: accepted for API compatibility; ignored.

    Returns:
        Deserialized Python object.
    """
    # Normalize Path -> str for extension checks
    if isinstance(file, Path):
        file = str(file)

    # Infer format from filename if not provided
    if file_format is None and isinstance(file, str):
        ext = Path(file).suffix.lower().lstrip(".")
        if ext in ("yaml", "yml"):
            file_format = "yaml"
        elif ext in ("json",):
            file_format = "json"
        elif ext in ("pickle", "pkl"):
            file_format = "pickle"
        else:
            raise TypeError(f"Unsupported or unknown file extension: .{ext}")

    # Normalize common aliases
    if file_format in ("yml",):
        file_format = "yaml"
    if file_format in ("pkl",):
        file_format = "pickle"

    if file_format not in ("json", "yaml", "pickle"):
        raise TypeError(f"Unsupported format: {file_format}")

    # Load from an open file-like object
    if hasattr(file, "read"):
        fobj = file  # trust caller’s mode
        if file_format == "json":
            if isinstance(fobj, io.BytesIO):
                fobj = io.TextIOWrapper(fobj, encoding="utf-8")
            return json.load(fobj, **kwargs)
        elif file_format == "yaml":
            if not _HAS_YAML:
                raise RuntimeError("PyYAML is not installed but YAML was requested.")
            if isinstance(fobj, io.BytesIO):
                fobj = io.TextIOWrapper(fobj, encoding="utf-8")
            return yaml.safe_load(fobj)
        else:  # pickle
            # ensure bytes interface
            if isinstance(fobj, io.TextIOBase):
                raise TypeError("Pickle must be loaded from a binary file-like (opened with 'rb').")
            return pickle.load(fobj)

    # Load from a path on local filesystem
    if isinstance(file, str):
        if file_format == "json":
            with open(file, "r", encoding="utf-8") as f:
                return json.load(f, **kwargs)
        elif file_format == "yaml":
            if not _HAS_YAML:
                raise RuntimeError("PyYAML is not installed but YAML was requested.")
            with open(file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        else:  # pickle
            with open(file, "rb") as f:
                return pickle.load(f)

    raise TypeError('"file" must be a filepath (str/Path) or a file-like object with .read().')


def _read_text(filename: str, encoding: str = "utf-8") -> str:
    """Read text from local path or http url, with basic compression by extension."""
    fn = str(filename)
    parsed = urlparse(fn)

    # Remote http or https
    if parsed.scheme in ("http", "https"):
        with urllib.request.urlopen(fn) as resp:
            data = resp.read()
        return data.decode(encoding)

    # Local filesystem
    path = Path(fn if parsed.scheme == "" else parsed.path)

    if fn.endswith(".gz"):
        with gzip.open(path, "rt", encoding=encoding) as f:
            return f.read()
    if fn.endswith(".bz2"):
        with bz2.open(path, "rt", encoding=encoding) as f:
            return f.read()
    if fn.endswith(".xz") or fn.endswith(".lzma"):
        with lzma.open(path, "rt", encoding=encoding) as f:
            return f.read()

    with open(path, "r", encoding=encoding) as f:
        return f.read()

def list_from_file(
    filename: str,
    prefix: str = "",
    offset: int = 0,
    max_num: int = 0,
    encoding: str = "utf-8",
    file_client_args: Optional[dict] = None,
    backend_args: Optional[dict] = None,
) -> List[str]:
    """Load a text file and return a list of lines with optional prefix and limit."""
    if file_client_args is not None:
        warnings.warn(
            '"file_client_args" is ignored in this mm free version. Use local or http paths.',
            DeprecationWarning,
        )
        if backend_args is not None:
            raise ValueError('"file_client_args" and "backend_args" cannot be set at the same time.')
    if backend_args is not None:
        warnings.warn(
            '"backend_args" is ignored in this mm free version. Use local or http paths.',
            DeprecationWarning,
        )

    text = _read_text(filename, encoding=encoding)

    cnt = 0
    item_list: List[str] = []
    with StringIO(text) as f:
        for _ in range(max(offset, 0)):
            if not f.readline():
                break
        for line in f:
            if 0 < max_num <= cnt:
                break
            item_list.append(prefix + line.rstrip("\n\r"))
            cnt += 1
    return item_list


def get_box_type(box_type: str) -> Tuple[Any, Any]:
    """Dummy replacement for get_box_type."""
    print(f'Boxes are not implemented. Use MMDetection3D if you need "{box_type}" boxes.')
    return None, None

class BoxCenterType3D(Tuple[float, float, float]):
    """Box center type of the dataset.

    Box center is the origin of the box coordinate system.
    This is passed as an origin argument to the `mmdet3d.structures`.

    Supported box center types:
    - KITTI: (0.5, 0.5, 0.0)
    - NUS: (0.5, 0.5, 0.5)
    """

    KITTI = (0.5, 0.5, 0.0)
    NUS = (0.5, 0.5, 0.5)
    
class LoadType3D(Enum):
    """Load type of the Det3D dataset.

    - FRAME: Load all of the instances in the frame.
    - MV_IMAGE: Load all of the instances in the frame and need
        to convert to the FOV-based data type to support image-based
        detector.
    - FOV_IMAGE: Only load the instances inside the default
        cam, and need to convert to the FOV-based data type to support
        image-based detector.
    - MV_LIDAR: Load all of the instances in the frame and need
        to convert to the BEV-based data type to support LiDAR-based
        detector. And the data will be packed into a multimodal format.
    """

    FRAME = "frame"
    MV_IMAGE = "mv_image"
    FOV_IMAGE = "fov_image"
    MV_LIDAR = "mv_lidar"


class Det3DDatasetMetaInfo:
    """
    Meta infos of the internal Det3DDatasets.

    We house the meta infos of the internal Det3DDatasets here.
    """

    BEVFrontStereo: ClassVar[Dict[str, Any]] = {
        "categories": {"car/van/suv": 0, "bus/truck/tram": 1, "bike": 2, "person": 3},
        "classes": ["car/van/suv", "bus/truck/tram", "bike", "person"],
        "palette": [[104, 163, 127], [39, 238, 54], [70, 23, 209], [123, 98, 174]],
        "dataset": "BEV_Dataset_Front_Stereo_Detection",
        "version": "v1.0",
    }

def collate_fn(batch):
    """Collate dict items like {'rgb': T, 'depth': T} while skipping Nones."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    out = {}
    # union of keys present in any sample
    keys = set().union(*(b.keys() for b in batch))
    for k in keys:
        vals = [b[k] for b in batch if (b is not None and k in b and b[k] is not None)]
        if not vals:
            continue
        if torch.is_tensor(vals[0]):
            out[k] = torch.stack(vals)
        else:
            out[k] = vals
    return out

def get_dist_info(group: Optional[dist.ProcessGroup] = None) -> Tuple[int, int]:
    """
    Return (rank, world_size) for the given process group.
    In a non-distributed environment (or before init), returns (0, 1).

    Args:
        group: Optional torch.distributed process group. If None, uses the default group.

    Returns:
        (rank, world_size)
    """
    # Not available or not initialized → act like single process
    if not (dist.is_available() and dist.is_initialized()):
        return 0, 1

    # Try to use the default group if none provided
    if group is None:
        try:
            group = dist.distributed_c10d._get_default_group()  # type: ignore[attr-defined]
        except Exception:
            group = None  # fall back to the global getters below

    try:
        if group is not None:
            rank = dist.get_rank(group)
            world_size = dist.get_world_size(group)
        else:
            # global default (works if a default group exists)
            rank = dist.get_rank()
            world_size = dist.get_world_size()
    except RuntimeError:
        # If something is off (e.g., default group not initialized yet), degrade gracefully
        return 0, 1

    return rank, world_size

class Compose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any] | None:
        for t in self.transforms:
            results = t.transform(results) if hasattr(t, "transform") else t(results)
            if results is None:  # mmdet semantics
                return None
        return results

def _build_pipeline(
    p: Union[None, Dict[str, Any], List[Any], DictConfig, ListConfig, Callable]
) -> Callable[[Dict], Optional[Dict]]:
    if p is None:
        return lambda x: x

    # Normalize OmegaConf containers to native Python, but KEEP objects intact
    if isinstance(p, (DictConfig, ListConfig)):
        p = OmegaConf.to_container(p, resolve=True)

    def maybe_build(x: Any):
        # Only instantiate dict-like configs that have _target_
        if isinstance(x, DictConfig):
            x = OmegaConf.to_container(x, resolve=True)
        if isinstance(x, dict):
            return instantiate(x) if "_target_" in x else x
        # Already a class instance or callable? pass through
        return x

    # dict with cfg_list
    if isinstance(p, dict) and "cfg_list" in p:
        cfgs = p["cfg_list"] or []
        return Compose([maybe_build(c) for c in cfgs])

    # plain list of configs/objects
    if isinstance(p, list):
        return Compose([maybe_build(c) for c in p])

    # already a callable (e.g., a prebuilt Compose)
    if callable(p):
        return p

    raise TypeError(f"Unsupported pipeline spec type after normalization: {type(p)}")
"""Aligned eight-channel image pipelines for RGB, infrared and depth."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
try:
    from mmdet.datasets.builder import PIPELINES
except ImportError:
    PIPELINES = None

def _find_stem(root: Path, stem: str) -> Path:
    matches = [p for p in root.iterdir() if p.is_file() and p.stem == stem]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one aligned file for {stem} in {root}, found {matches}")
    return matches[0]


def resolve_rgb_path(rgb_path: Path) -> Path:
    if rgb_path.is_file():
        return rgb_path
    matches = [rgb_path.with_suffix(suffix) for suffix in (".jpg", ".jpeg", ".png", ".bmp") if rgb_path.with_suffix(suffix).is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one RGB file for {rgb_path.stem}, found {matches}")
    return matches[0]


def build_eight_channel(rgb_path: Path, infrared_root: Path, depth_root: Path) -> np.ndarray:
    rgb_path = resolve_rgb_path(rgb_path)
    infrared_path = _find_stem(infrared_root, rgb_path.stem)
    depth_path = _find_stem(depth_root, rgb_path.stem)
    try:
        with Image.open(rgb_path) as handle:
            rgb = np.asarray(handle.convert("RGB"))[:, :, ::-1].copy()
        with Image.open(infrared_path) as handle:
            infrared = np.asarray(handle)
        with Image.open(depth_path) as handle:
            depth = np.asarray(handle)
    except Exception as exc:
        raise RuntimeError(f"Cannot decode aligned sample {rgb_path.name}: {exc}") from exc
    if infrared.ndim == 3:
        infrared = cv2.cvtColor(infrared, cv2.COLOR_BGR2GRAY)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if rgb.shape[:2] != infrared.shape[:2] or rgb.shape[:2] != depth.shape[:2]:
        raise ValueError(
            f"Aligned shape mismatch for {rgb_path.name}: rgb={rgb.shape}, "
            f"infrared={infrared.shape}, depth={depth.shape}"
        )

    rgb = rgb.astype(np.float32) / 255.0
    infrared = infrared.astype(np.float32)
    if infrared.max(initial=0.0) > 1.0:
        infrared /= 255.0 if infrared.max() <= 255.0 else max(float(infrared.max()), 1.0)
    infrared = infrared[..., None]

    metric = depth.dtype == np.uint16 or depth.max(initial=0) > 255
    depth_float = depth.astype(np.float32)
    valid = np.isfinite(depth_float) & (depth_float >= 300.0) & (depth_float <= 20000.0)
    if not metric:
        valid = np.isfinite(depth_float) & (depth_float > 0.0)
        depth_linear = np.clip(depth_float / 255.0, 0.0, 1.0)
    else:
        clipped = np.clip(depth_float, 300.0, 20000.0)
        depth_linear = (clipped - 300.0) / 19700.0
    depth_linear = np.where(valid, depth_linear, 0.0).astype(np.float32)
    inverse_depth = np.where(valid, 1.0 - depth_linear, 0.0).astype(np.float32)
    valid_mask = valid.astype(np.float32)
    metric_flag = np.full(valid_mask.shape, 1.0 if metric else 0.0, dtype=np.float32)
    return np.concatenate(
        [rgb, infrared, depth_linear[..., None], inverse_depth[..., None], valid_mask[..., None], metric_flag[..., None]],
        axis=2,
    )


def resize_eight_channel(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    output = np.empty((height, width, 8), dtype=np.float32)
    for index in range(8):
        interpolation = cv2.INTER_NEAREST if index in (6, 7) else cv2.INTER_LINEAR
        output[..., index] = cv2.resize(image[..., index], (width, height), interpolation=interpolation)
    return output


if PIPELINES is not None:

    @PIPELINES.register_module()
    class LoadAlignedMultimodal:
        def __init__(self, infrared_root: str, depth_root: str):
            self.infrared_root = Path(infrared_root)
            self.depth_root = Path(depth_root)

        def __call__(self, results):
            filename_value = results.get("filename")
            if filename_value is None:
                filename_value = results.get("img_info", {}).get("filename")
            if filename_value is None:
                raise KeyError("filename")
            filename = Path(filename_value)
            if not filename.is_absolute() and results.get("img_prefix"):
                filename = Path(results["img_prefix"]) / filename
            results["filename"] = str(filename)
            results.setdefault("ori_filename", filename.name)
            results["img"] = build_eight_channel(filename, self.infrared_root, self.depth_root)
            results["img_shape"] = results["img"].shape
            results["ori_shape"] = results["img"].shape
            results["img_fields"] = ["img"]
            return results

        def __repr__(self):
            return f"{self.__class__.__name__}(infrared_root={self.infrared_root}, depth_root={self.depth_root})"

    @PIPELINES.register_module()
    class ResizeAlignedMultimodal:
        def __init__(self, img_scale=None, keep_ratio=True, multiscale_mode="value", **kwargs):
            self.img_scale = img_scale
            self.keep_ratio = keep_ratio
            self.multiscale_mode = multiscale_mode

        def __call__(self, results):
            height, width = results["img"].shape[:2]
            scale_value = self.img_scale or results.get("scale")
            if scale_value is None:
                raise ValueError("ResizeAlignedMultimodal requires img_scale")
            if isinstance(scale_value, (list, tuple)) and scale_value and isinstance(scale_value[0], (list, tuple)):
                scale_value = scale_value[np.random.randint(len(scale_value))]
            target_width, target_height = scale_value
            if self.keep_ratio:
                scale = min(target_width / width, target_height / height)
                target_width, target_height = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
            results["img"] = resize_eight_channel(results["img"], (target_width, target_height))
            scale_x, scale_y = target_width / width, target_height / height
            results["img_shape"] = results["img"].shape
            results["scale"] = (target_width, target_height)
            results["scale_factor"] = np.array([scale_x, scale_y, scale_x, scale_y], dtype=np.float32)
            if "gt_bboxes" in results:
                results["gt_bboxes"] = results["gt_bboxes"] * results["scale_factor"]
            return results

    @PIPELINES.register_module()
    class NormalizeAlignedMultimodal:
        def __init__(self, mean, std, to_rgb=True):
            self.mean = np.asarray(mean, dtype=np.float32)
            self.std = np.asarray(std, dtype=np.float32)
            self.to_rgb = bool(to_rgb)

        def __call__(self, results):
            image = results["img"]
            if self.to_rgb:
                image = image[..., [2, 1, 0, 3, 4, 5, 6, 7]]
            image = image.copy()
            image[..., :3] = (image[..., :3] * 255.0 - self.mean) / self.std
            results["img"] = image
            results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
            return results


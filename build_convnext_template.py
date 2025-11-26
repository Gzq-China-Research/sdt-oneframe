#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_convnext_template.py (v3)
- 基于术前带贴片图像 + 旧模板目录(meta.json)
- 裁剪贴片邻域 patch
- 使用 ConvNeXt 提取全局模板特征向量
- 存成单独的 meta_convnext.json，不修改旧 meta.json

兼容你当前的 meta.json 格式，例如：
{
  "circle_center_preop": [cx, cy],
  "circle_radius_px": 7.22,
  "preop_image_size": [W, H],
  "rel_center_image": [rx, ry],
  ...
}
"""

import json
import argparse
from pathlib import Path

import cv2
import numpy as np

from convnext_utils import ConvNeXtFeatureExtractor


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _parse_center(entry):
    """
    将 meta 中的 circle_center_preop / circle_center 解析为 (cx, cy)
    兼容格式：
    1) {"cx": 123, "cy": 456}
    2) [123, 456] 或 (123, 456)
    3) [{"cx": 123, "cy": 456}, ...]
    """
    if entry is None:
        raise KeyError("center entry is None")

    # dict 形式 {"cx":..., "cy":...}
    if isinstance(entry, dict):
        if "cx" in entry and "cy" in entry:
            return float(entry["cx"]), float(entry["cy"])
        if "x" in entry and "y" in entry:
            return float(entry["x"]), float(entry["y"])
        raise KeyError(f"Unsupported center dict format: {entry}")

    # list / tuple 形式
    if isinstance(entry, (list, tuple)):
        # [cx, cy]
        if len(entry) >= 2 and isinstance(entry[0], (int, float)) and isinstance(entry[1], (int, float)):
            return float(entry[0]), float(entry[1])

        # [{"cx":...,"cy":...}, ...]
        if len(entry) > 0 and isinstance(entry[0], dict):
            first = entry[0]
            if "cx" in first and "cy" in first:
                return float(first["cx"]), float(first["cy"])
            if "x" in first and "y" in first:
                return float(first["x"]), float(first["y"])

    raise TypeError(f"Unsupported circle center format: type={type(entry)}, value={entry}")


def _parse_radius(entry, default_val=40.0):
    """
    将 meta 中的 circle_radius_px / circle_radius 等解析为 float 半径
    兼容：
    1) 纯数字
    2) [r] / (r, ...)
    3) {"r": xx}
    """
    if entry is None:
        return float(default_val)

    # 纯数字
    if isinstance(entry, (int, float)):
        return float(entry)

    # dict
    if isinstance(entry, dict):
        if "r" in entry:
            return float(entry["r"])
        if "radius" in entry:
            return float(entry["radius"])
        for k in ["radius_px", "r_px"]:
            if k in entry:
                return float(entry[k])
        # 实在没有就用默认
        return float(default_val)

    # list / tuple
    if isinstance(entry, (list, tuple)):
        if len(entry) > 0 and isinstance(entry[0], (int, float)):
            return float(entry[0])
        if len(entry) > 0 and isinstance(entry[0], dict):
            first = entry[0]
            return _parse_radius(first, default_val)

    return float(default_val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preop", required=True, help="术前带贴片图像路径")
    ap.add_argument(
        "--templates",
        required=True,
        help="旧模板目录 (内含 meta.json，由 build_from_one.py 生成)",
    )
    ap.add_argument(
        "--out_meta",
        required=True,
        help="输出 ConvNeXt 模板 meta 路径，如 work/templates/meta_convnext.json",
    )
    ap.add_argument(
        "--model_name",
        default="convnext_tiny",
        choices=["convnext_tiny", "convnext_small"],
        help="ConvNeXt backbone 名称",
    )
    ap.add_argument(
        "--img_size",
        type=int,
        default=384,
        help="ConvNeXt 输入大小 (宽高相同)",
    )
    ap.add_argument(
        "--patch_scale",
        type=float,
        default=4.0,
        help="贴片邻域 patch 边长 = patch_scale * circle_radius_px",
    )
    ap.add_argument(
        "--device",
        default=None,
        help='"cuda" / "cpu" / None(自动选择)',
    )
    args = ap.parse_args()

    preop_path = Path(args.preop)
    tmpl_dir = Path(args.templates)
    out_meta_path = Path(args.out_meta)

    if not preop_path.is_file():
        raise FileNotFoundError(str(preop_path))
    if not (tmpl_dir / "meta.json").is_file():
        raise FileNotFoundError(f"{tmpl_dir/'meta.json'} not found. 请先运行 build_from_one.py")

    # 1) 读术前图像
    pre_bgr = cv2.imread(str(preop_path), cv2.IMREAD_COLOR)
    if pre_bgr is None:
        raise FileNotFoundError(str(preop_path))
    H, W, _ = pre_bgr.shape

    # 2) 读取旧 meta.json
    meta_old = json.load(open(tmpl_dir / "meta.json", "r", encoding="utf-8"))

    # 2.1 贴片中心
    center_entry = None
    if "circle_center_preop" in meta_old:
        center_entry = meta_old["circle_center_preop"]
    elif "circle_center" in meta_old:
        center_entry = meta_old["circle_center"]
    elif "circle" in meta_old:
        center_entry = meta_old["circle"]
    else:
        raise KeyError("meta.json 中未找到 circle_center_preop / circle_center / circle 等字段。")

    cx, cy = _parse_center(center_entry)

    # 2.2 半径
    radius_entry = None
    if "circle_radius_px" in meta_old:
        radius_entry = meta_old["circle_radius_px"]
    elif "circle_radius" in meta_old:
        radius_entry = meta_old["circle_radius"]
    elif "circle" in meta_old and isinstance(meta_old["circle"], dict):
        radius_entry = meta_old["circle"].get("r", None)
    elif "patch_radius_px" in meta_old:
        radius_entry = meta_old["patch_radius_px"]

    radius_px = _parse_radius(radius_entry, default_val=40.0)

    # 2.3 术前图像尺寸（旧 meta 是 [W,H]，我们这里存成 dict）
    pre_size_old = meta_old.get("preop_image_size", None)
    if isinstance(pre_size_old, (list, tuple)) and len(pre_size_old) >= 2:
        W_pre = int(pre_size_old[0])
        H_pre = int(pre_size_old[1])
    elif isinstance(pre_size_old, dict) and "width" in pre_size_old and "height" in pre_size_old:
        W_pre = int(pre_size_old["width"])
        H_pre = int(pre_size_old["height"])
    else:
        # 兜底用真实图像尺寸
        W_pre = W
        H_pre = H

    # 2.4 rel_center_image（旧 meta 是 [rx,ry]）
    rel_old = meta_old.get("rel_center_image", None)
    if isinstance(rel_old, (list, tuple)) and len(rel_old) >= 2:
        rx_old = float(rel_old[0])
        ry_old = float(rel_old[1])
    elif isinstance(rel_old, dict):
        rx_old = float(rel_old.get("rx", cx / max(W_pre, 1)))
        ry_old = float(rel_old.get("ry", cy / max(H_pre, 1)))
    else:
        # 没有的话自己用中心算一个
        rx_old = cx / max(W_pre, 1)
        ry_old = cy / max(H_pre, 1)

    # 3) 按贴片中心裁 patch
    L = int(max(32, args.patch_scale * radius_px))  # 最小边长 32 防止太小
    x0 = int(np.clip(cx - L // 2, 0, W - 1))
    y0 = int(np.clip(cy - L // 2, 0, H - 1))
    x1 = int(np.clip(x0 + L, 1, W))
    y1 = int(np.clip(y0 + L, 1, H))
    patch = pre_bgr[y0:y1, x0:x1, :].copy()
    if patch.size == 0:
        raise RuntimeError(
            f"提取 ConvNeXt 模板 patch 失败，得到空区域，"
            f"cx={cx}, cy={cy}, L={L}, W={W}, H={H}"
        )

    # 4) 用 ConvNeXt 提取模板特征
    extractor = ConvNeXtFeatureExtractor(
        model_name=args.model_name,
        img_size=args.img_size,
        device=args.device,
    )
    f_tpl = extractor.extract_global_feature(patch)  # [C]

    # 5) 组织新的 ConvNeXt meta（这里我们把中心/尺寸/rel_center 统一写成 dict）
    conv_meta = {
        "model_name": args.model_name,
        "img_size": int(args.img_size),
        "patch_scale": float(args.patch_scale),
        "circle_center_preop": {"cx": float(cx), "cy": float(cy)},
        "circle_radius_px": float(radius_px),
        "preop_image_size": {"width": int(W_pre), "height": int(H_pre)},
        "rel_center_image": {
            "rx": float(rx_old),
            "ry": float(ry_old),
        },
        "template_feature": f_tpl.tolist(),  # ConvNeXt 模板向量
    }

    ensure_dir(out_meta_path.parent)
    with open(out_meta_path, "w", encoding="utf-8") as f:
        json.dump(conv_meta, f, indent=2, ensure_ascii=False)

    print(f"[build_convnext_template] conv_meta saved -> {out_meta_path}")
    print(
        f"  center = ({cx:.1f}, {cy:.1f}), radius_px = {radius_px:.2f}, "
        f"L = {L}, preop_size = ({W_pre}x{H_pre}), rel_center = ({rx_old:.3f},{ry_old:.3f})"
    )


if __name__ == "__main__":
    main()

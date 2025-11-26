#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_convnext_template.py (v4 - Context Aware)
- 强制使用大尺寸 Patch (如 192px) 包含解剖纹理上下文
- 解决小贴片导致特征模糊的问题
"""

import json
import argparse
from pathlib import Path
import cv2
import numpy as np
from convnext_utils import ConvNeXtFeatureExtractor


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preop", required=True, help="术前带贴片图像路径")
    ap.add_argument("--templates", required=True, help="旧模板目录 (含 meta.json)")
    ap.add_argument("--out_meta", required=True, help="输出 meta_convnext.json 路径")
    ap.add_argument("--model_name", default="convnext_tiny")
    ap.add_argument("--context_size", type=int, default=192, help="强制 Patch 物理尺寸 (像素)，建议 192 或 224")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    preop_path = Path(args.preop)
    tmpl_dir = Path(args.templates)
    out_meta_path = Path(args.out_meta)

    # 1) 读取术前图和旧 meta
    if not preop_path.is_file(): raise FileNotFoundError(str(preop_path))
    pre_bgr = cv2.imread(str(preop_path), cv2.IMREAD_COLOR)
    H, W = pre_bgr.shape[:2]

    meta_old = json.load(open(tmpl_dir / "meta.json", "r", encoding="utf-8"))

    # 解析中心点 (兼容多种格式)
    c_entry = meta_old.get("circle_center_preop", meta_old.get("circle_center"))
    if isinstance(c_entry, dict):
        cx, cy = float(c_entry.get("cx", c_entry.get("x"))), float(c_entry.get("cy", c_entry.get("y")))
    else:
        cx, cy = float(c_entry[0]), float(c_entry[1])

    # 解析相对坐标 (用于粗搜先验)
    rel_entry = meta_old.get("rel_center_image", None)
    if rel_entry:
        if isinstance(rel_entry, dict):
            rx, ry = float(rel_entry.get("rx", 0.5)), float(rel_entry.get("ry", 0.5))
        else:
            rx, ry = float(rel_entry[0]), float(rel_entry[1])
    else:
        rx, ry = cx / W, cy / H

    # 2) 裁切包含 Context 的大 Patch
    # 不再使用 circle_radius_px * scale，直接用物理尺寸
    L = args.context_size
    half = L // 2

    x_center_int = int(round(cx))
    y_center_int = int(round(cy))

    # 计算边界，使用 Padding 策略防止越界
    x0 = x_center_int - half
    y0 = y_center_int - half

    # 计算需要 Pad 多少
    pad_top = max(0, -y0)
    pad_bottom = max(0, y0 + L - H)
    pad_left = max(0, -x0)
    pad_right = max(0, x0 + L - W)

    # 对原图做 Padding (黑色填充)
    img_padded = cv2.copyMakeBorder(pre_bgr, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,
                                    value=(0, 0, 0))

    # 在 Pad 后的图上裁切
    x0_new = x0 + pad_left
    y0_new = y0 + pad_top
    patch = img_padded[y0_new: y0_new + L, x0_new: x0_new + L].copy()

    if patch.shape[0] != L or patch.shape[1] != L:
        print(f"WARN: Patch shape mismatch {patch.shape}, expected {L}x{L}")
        patch = cv2.resize(patch, (L, L))

    # 3) 提取特征
    extractor = ConvNeXtFeatureExtractor(model_name=args.model_name, img_size=224, device=args.device)
    # 使用 batch 接口提特征 (输入 list)
    f_tpl = extractor.extract_batch_features([patch])[0]

    # 4) 保存结果
    conv_meta = {
        "model_name": args.model_name,
        "patch_context_size": L,  # 记录这个关键参数
        "circle_center_preop": {"cx": cx, "cy": cy},
        "preop_image_size": {"width": W, "height": H},
        "rel_center_image": {"rx": rx, "ry": ry},
        "template_feature": f_tpl.tolist()
    }

    ensure_dir(out_meta_path.parent)
    with open(out_meta_path, "w", encoding="utf-8") as f:
        json.dump(conv_meta, f, indent=2, ensure_ascii=False)

    print(f"[build_convnext] Patch size: {L}x{L}, Center: ({cx:.1f}, {cy:.1f})")
    print(f"[build_convnext] Meta saved to {out_meta_path}")


if __name__ == "__main__":
    main()
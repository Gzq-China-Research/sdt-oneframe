#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sdt_oneframe_cli_convnext.py (v3.1 - Fix for Context-Aware Pipeline)
- 适配新的 build_convnext_template.py (移除 img_size, 新增 context_size)
- 适配新的 coarse_search_convnext.py
"""

import argparse
import subprocess
from pathlib import Path
import sys

def run(cmd_list):
    print(">>>", " ".join(cmd_list))
    subprocess.check_call(cmd_list)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preop", required=True, help="术前带贴片图像路径")
    ap.add_argument("--intra", required=True, help="术中无贴片图像路径")
    ap.add_argument(
        "--outdir",
        required=True,
        help="最终结果目录，如 results_chest_convnext",
    )
    ap.add_argument(
        "--params",
        default="configs/params_convnext.yaml",
        help="ConvNeXt 粗搜参数 YAML",
    )
    ap.add_argument(
        "--shape_prior",
        default=None,
        help="形状先验 JSON（如有）",
    )
    ap.add_argument(
        "--templates_dir",
        default="work/templates",
        help="模板输出目录 (由 build_from_one.py 使用)",
    )
    ap.add_argument(
        "--anatomy",
        choices=["auto", "neck", "chest", "groin"],
        default="chest",
    )
    ap.add_argument("--roi", default=None, help="可选 ROI JSON（本版本暂未使用）")
    ap.add_argument(
        "--patch_diam_mm",
        type=float,
        default=10.0,
        help="圆形贴片直径 (mm)",
    )
    ap.add_argument(
        "--patch_height_mm",
        type=float,
        default=0.39,
        help="圆形贴片高度 (mm)",
    )
    ap.add_argument(
        "--mask_inner_ratio",
        type=float,
        default=0.9,
        help="贴片内部遮挡的半径比例",
    )
    ap.add_argument(
        "--debug_dir",
        default=None,
        help="调试输出目录",
    )
    args = ap.parse_args()

    preop = args.preop
    intra = args.intra
    outdir = Path(args.outdir)
    templates_dir = Path(args.templates_dir)
    params_yaml = args.params
    anatomy = args.anatomy

    outdir.mkdir(parents=True, exist_ok=True)
    if args.debug_dir:
        Path(args.debug_dir).mkdir(parents=True, exist_ok=True)

    # 1) 旧 build_from_one：检测贴片 + 生成经典模板/meta.json
    # 注意：这里依然使用 Python 解释器调用，确保环境一致
    cmd1 = [
        sys.executable,
        "build_from_one.py",
        "--preop",
        preop,
        "--out",
        str(templates_dir),
        "--patch_diam_mm",
        str(args.patch_diam_mm),
        "--patch_height_mm",
        str(args.patch_height_mm),
        "--mask_inner_ratio",
        str(args.mask_inner_ratio),
        "--anatomy",
        anatomy,
        "--params",
        "configs/params.yaml",  # 基础模板参数
    ]
    if args.debug_dir:
        cmd1 += ["--debug_dir", str(Path(args.debug_dir) / "preop")]
    run(cmd1)

    # 2) 新 build_convnext_template：基于术前 + meta.json 生成 ConvNeXt 模板
    # 修改点：移除 --img_size，改为 --context_size 192
    conv_meta_path = templates_dir / "meta_convnext.json"
    cmd2 = [
        sys.executable,
        "build_convnext_template.py",
        "--preop",
        preop,
        "--templates",
        str(templates_dir),
        "--out_meta",
        str(conv_meta_path),
        "--model_name",
        "convnext_tiny",
        "--context_size",  # <--- NEW: 强制大感受野
        "192",             # 像素，建议 192 或 224
    ]
    run(cmd2)

    # 3) 新 coarse_search_convnext：术中 ConvNeXt 粗搜 -> candidates_convnext.json
    cand_path = templates_dir / "candidates_convnext.json"
    cmd3 = [
        sys.executable,
        "coarse_search_convnext.py",
        "--intra",
        intra,
        "--templates",
        str(templates_dir),
        "--conv_meta",
        str(conv_meta_path),
        "--out",
        str(cand_path),
        "--params",
        params_yaml,
        "--anatomy",
        anatomy,
    ]
    if args.debug_dir:
        cmd3 += ["--debug_dir", str(Path(args.debug_dir) / "coarse_convnext")]
    run(cmd3)

    # 4) 新 fine_match_convnext：吃 ConvNeXt 生成的候选，做细配
    relocalize_json = outdir / "relocalize_convnext.json"
    vis_png = outdir / "vis_relocalize_convnext.png"
    cmd4 = [
        sys.executable,
        "fine_match_convnext.py",
        "--intra",
        intra,
        "--templates",
        str(templates_dir),
        "--candidates",
        str(cand_path),
        "--out",
        str(relocalize_json),
        "--vis",
        str(vis_png),
        "--anatomy",
        anatomy,
        "--params",
        "configs/params.yaml",  # 细配参数
    ]
    if args.debug_dir:
        cmd4 += ["--debug_dir", str(Path(args.debug_dir) / "fine_convnext")]
    run(cmd4)

    print("Done.")
    print(f"- Candidates   : {cand_path}")
    print(f"- Relocalize   : {relocalize_json}")
    print(f"- Visualization: {vis_png}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sdt_oneframe_cli.py (v3.9)
- 支持 --anatomy {auto, neck, chest, groin}
- 统一把 --anatomy、--params、--roi、--debug_dir 等透传到三个子程序
- Windows/Linux 通用
"""

import argparse
import subprocess
from pathlib import Path
import os
import sys

def run(cmd):
    print(">>>", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preop", required=True, help="术前：带贴片图像路径")
    ap.add_argument("--intra", required=True, help="术中：无贴片图像路径")
    ap.add_argument("--outdir", default="results", help="最终结果输出目录（细配JSON/可视化）")
    ap.add_argument("--params", default="configs/params.yaml", help="参数文件")
    ap.add_argument("--shape_prior", default=None, help="解剖先验 JSON（可选）")
    ap.add_argument("--templates_dir", default="work/templates", help="模板库输出目录")
    ap.add_argument("--roi", default=None, help="术中 ROI json（可选）")
    ap.add_argument("--patch_diam_mm", type=float, default=10.0)
    ap.add_argument("--patch_height_mm", type=float, default=0.39)
    ap.add_argument("--mask_inner_ratio", type=float, default=0.90)
    ap.add_argument("--anatomy", choices=["auto","neck","chest","groin"], default="auto",
                    help="解剖区域（neck/chest/groin 或 auto）")
    ap.add_argument("--debug_dir", default=None, help="调试输出根目录（可选）")
    args = ap.parse_args()

    # 路径准备
    outdir = Path(args.outdir)
    tmpldir = Path(args.templates_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    tmpldir.mkdir(parents=True, exist_ok=True)

    # 子路径
    candidates_json = str(tmpldir / "candidates.json")
    vis_png         = str(outdir / "vis_relocalize.png")
    relocalize_json = str(outdir / "relocalize.json")

    # 1) 术前模板生成
    cmd1 = [
        sys.executable, "build_from_one.py",
        "--preop", args.preop,
        "--out", str(tmpldir),
        "--patch_diam_mm", str(args.patch_diam_mm),
        "--patch_height_mm", str(args.patch_height_mm),
        "--mask_inner_ratio", str(args.mask_inner_ratio),
        "--anatomy", args.anatomy,
        "--params", args.params,
    ]
    if args.shape_prior:
        cmd1 += ["--shape_prior", args.shape_prior]
    if args.debug_dir:
        cmd1 += ["--debug_dir", str(Path(args.debug_dir) / "preop")]

    run(cmd1)

    # 2) 粗配
    cmd2 = [
        sys.executable, "coarse_search.py",
        "--intra", args.intra,
        "--templates", str(tmpldir),
        "--out", candidates_json,
        "--params", args.params,
        "--anatomy", args.anatomy,
    ]
    if args.roi:
        cmd2 += ["--roi", args.roi]
    if args.debug_dir:
        cmd2 += ["--debug_dir", str(Path(args.debug_dir) / "coarse")]

    run(cmd2)

    # 3) 细配
    cmd3 = [
        sys.executable, "fine_match.py",
        "--intra", args.intra,
        "--templates", str(tmpldir),
        "--candidates", candidates_json,
        "--out", relocalize_json,
        "--vis", vis_png,
        "--anatomy", args.anatomy,
        "--params", args.params,
    ]
    if args.debug_dir:
        cmd3 += ["--debug_dir", str(Path(args.debug_dir) / "fine")]

    run(cmd3)

    print("\nDone.")
    print(f"- Candidates   : {candidates_json}")
    print(f"- Relocalize   : {relocalize_json}")
    print(f"- Visualization: {vis_png}")

if __name__ == "__main__":
    main()

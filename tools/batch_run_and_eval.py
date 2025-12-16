#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/batch_run_and_eval.py
批量实验脚本：适配 subject/part/view_xx 目录结构
"""

import os
import json
import pandas as pd
import numpy as np
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm


def main():
    # 你的数据根目录
    data_root = "data/real_human"
    # 实验结果保存位置
    results_root = "experiments/exp_01_baseline"

    script_path = "sdt_oneframe_cli_convnext.py"
    params_yaml = "configs/params_convnext.yaml"

    # 扫描所有含 gt.json 的子目录
    cases = []
    for root, dirs, files in os.walk(data_root):
        if "gt.json" in files:
            cases.append(Path(root))

    print(f"发现 {len(cases)} 组已标注数据。准备开始...")

    records = []

    for idx, case_dir in tqdm(enumerate(cases), total=len(cases), desc="Processing"):
        gt_path = case_dir / "gt.json"
        try:
            gt_data = json.load(open(gt_path, 'r'))
        except:
            print(f"无法读取 {gt_path}, 跳过")
            continue

        # 路径解析：subject_01 / neck / view_01
        view_name = case_dir.name  # view_01
        part_name = case_dir.parent.name  # neck
        subj_name = case_dir.parent.parent.name  # subject_01

        # 构造输出ID
        case_id = f"{subj_name}_{part_name}_{view_name}"
        out_dir = Path(results_root) / case_id

        # 获取图片绝对路径
        pre_img = str(case_dir / Path(gt_data["pre_path"]).name)
        intra_img = str(case_dir / Path(gt_data["intra_path"]).name)

        # 运行算法
        cmd = [
            sys.executable, script_path,
            "--preop", pre_img,
            "--intra", intra_img,
            "--outdir", str(out_dir),
            "--params", params_yaml,
            "--anatomy", "auto",  # 也可以这里改成 part_name 如果你的命名是标准的
            "--patch_diam_mm", "10.0",
            # "--debug_dir", str(out_dir / "debug") # 需要debug图时解开
        ]

        try:
            # capture_output=True 让它别刷屏，只在出错时显示
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"\nCase {case_id} 运行失败: {e.stderr.decode()}")
            continue

        # 读取结果与评估
        res_json = out_dir / "relocalize_convnext.json"
        if not res_json.exists(): continue

        pred_data = json.load(open(res_json, 'r'))

        gt_xy = np.array(gt_data["center_gt"])
        pred_xy = np.array(pred_data["center_img"])
        error_px = np.linalg.norm(gt_xy - pred_xy)

        # 记录到表格
        rec = {
            "Subject": subj_name,
            "Part": part_name,
            "View": view_name,
            "Error_px": round(error_px, 2),
            "Score": round(pred_data.get("score", 0), 4),
            "Fallback": pred_data.get("fallback", False),
            "CycleErr": round(pred_data.get("cycle_err", -1), 2)
        }
        records.append(rec)

    # 生成 CSV 报告
    if records:
        df = pd.DataFrame(records)
        Path(results_root).mkdir(parents=True, exist_ok=True)
        csv_path = Path(results_root) / "final_report.csv"
        df.to_csv(csv_path, index=False)

        print("\n" + "=" * 30)
        print(f"测试完成！共 {len(df)} 组数据。")
        print(f"平均误差: {df['Error_px'].mean():.2f} px")
        print(f"成功率 (<10px): {(df['Error_px'] < 10).mean() * 100:.1f}%")
        print(f"报告已保存: {csv_path}")
    else:
        print("未产生任何结果。")


if __name__ == "__main__":
    main()
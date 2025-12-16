#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/launcher_labeling.py
自动扫描数据目录，依次启动标注工具。
解放双手，不需要每次都手动输入长路径。
"""

import os
import sys
import subprocess
from pathlib import Path


def main():
    # 指向你的数据根目录
    data_root = Path("data/real_human")
    label_script = Path("tools/label_ground_truth.py")

    if not data_root.exists():
        print(f"错误：找不到数据目录 {data_root}")
        return

    # 查找所有的 view_xx 文件夹
    # 结构: subject/part/view/
    tasks = []
    for subj in sorted(data_root.iterdir()):
        if not subj.is_dir(): continue
        for part in sorted(subj.iterdir()):
            if not part.is_dir(): continue
            for view in sorted(part.iterdir()):
                if not view.is_dir(): continue

                # 检查是否已标注
                if (view / "gt.json").exists():
                    continue  # 跳过已完成的

                # 检查是否存在图片 (兼容 bmp, jpg, png)
                pre_img = None
                intra_img = None

                for ext in [".bmp", ".jpg", ".png"]:
                    if (view / f"pre{ext}").exists(): pre_img = view / f"pre{ext}"
                    if (view / f"intra{ext}").exists(): intra_img = view / f"intra{ext}"

                if pre_img and intra_img:
                    tasks.append((pre_img, intra_img))

    print(f"扫描完成：发现 {len(tasks)} 组待标注数据。")
    if len(tasks) == 0:
        print("太棒了！所有数据都已标注完成。")
        return

    print("即将开始连续标注... (按 Ctrl+C 可随时终止)")
    print("-" * 40)

    for i, (pre, intra) in enumerate(tasks):
        print(f"[{i + 1}/{len(tasks)}] 正在标注: {pre.parent}")

        # 调用 label_ground_truth.py
        cmd = [
            sys.executable, str(label_script),
            "--pre", str(pre),
            "--intra", str(intra)
        ]

        try:
            # 运行子进程，等待标注窗口关闭后再继续下一个
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError:
            print("标注中断或出错。")
            break
        except KeyboardInterrupt:
            print("\n用户手动停止。")
            break

    print("\n本轮标注结束。请重新运行此脚本检查是否全部完成。")


if __name__ == "__main__":
    main()
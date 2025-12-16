#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/label_ground_truth.py (增强交互版)
- 修复了按键连击导致第二张图直接跳过的问题。
- 增加了点数强制检查：点不够，按回车也没用。
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path

# 全局变量
points = []
img_display = None
img_raw = None  # 保留原始图，用于重绘


def on_mouse(event, x, y, flags, param):
    global points, img_display, img_raw

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        # 实时反馈：画圈和数字
        cv2.circle(img_display, (x, y), 4, (0, 0, 255), -1)
        cv2.putText(img_display, str(len(points)), (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Labeling", img_display)

    elif event == cv2.EVENT_RBUTTONDOWN:
        # 右键撤销上一个点
        if len(points) > 0:
            points.pop()
            # 重绘图片
            img_display = img_raw.copy()
            for i, pt in enumerate(points):
                cv2.circle(img_display, pt, 4, (0, 0, 255), -1)
                cv2.putText(img_display, str(i + 1), (pt[0] + 5, pt[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Labeling", img_display)


def interaction_loop(window_name, target_count, desc_text):
    """
    强制循环，直到用户点够了数量并按下了确认键
    """
    global points

    print(f"\n--- {desc_text} ---")
    print(f"目标: 点击 {target_count} 个点。")
    print("操作: [左键] 点击, [右键] 撤销, [空格/回车] 确认完成")

    while True:
        cv2.imshow(window_name, img_display)
        # 等待按键 (10ms 刷新一次)
        key = cv2.waitKey(10) & 0xFF

        # 按下 空格(32) 或 回车(13)
        if key == 32 or key == 13:
            if len(points) == target_count:
                print(f"  -> 已确认 {len(points)} 个点。")
                break
            else:
                print(f"  [警告] 还需要点击 {target_count} 个点，当前只有 {len(points)} 个！无法继续。")

        # 按下 ESC(27) 退出
        if key == 27:
            print("用户取消。")
            exit(0)


def main():
    global points, img_display, img_raw
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True)
    ap.add_argument("--intra", required=True)
    args = ap.parse_args()

    pre_path = Path(args.pre)
    intra_path = Path(args.intra)

    img_pre = cv2.imread(str(pre_path))
    img_intra = cv2.imread(str(intra_path))

    if img_pre is None or img_intra is None:
        print("错误：无法读取图片。请检查路径或格式(bmp/jpg)。")
        return

    # 窗口初始化
    cv2.namedWindow("Labeling", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Labeling", 1024, 768)
    cv2.setMouseCallback("Labeling", on_mouse)

    # ----------------------
    # 阶段 1: 术前图 (5点)
    # ----------------------
    points = []
    img_raw = img_pre.copy()
    img_display = img_pre.copy()

    interaction_loop("Labeling", 5, "阶段 1/2: 术前图 (1个中心 + 4个外围)")

    # 保存数据
    center_pre = np.array([points[0]], dtype=np.float32)
    markers_pre = np.array(points[1:], dtype=np.float32)

    # *** 关键修复：切换图片间隙，销毁窗口防止缓存 ***
    cv2.destroyWindow("Labeling")
    cv2.waitKey(100)  # 给系统一点喘息时间

    # 重建窗口
    cv2.namedWindow("Labeling", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Labeling", 1024, 768)
    cv2.setMouseCallback("Labeling", on_mouse)

    # ----------------------
    # 阶段 2: 术中图 (4点)
    # ----------------------
    points = []
    img_raw = img_intra.copy()
    img_display = img_intra.copy()

    interaction_loop("Labeling", 4, "阶段 2/2: 术中图 (4个对应的外围点)")

    markers_intra = np.array(points, dtype=np.float32)

    # ----------------------
    # 计算与保存
    # ----------------------
    H, mask = cv2.findHomography(markers_pre, markers_intra, cv2.RANSAC, 5.0)

    # 映射中心
    center_gt_3d = cv2.perspectiveTransform(center_pre.reshape(1, 1, 2), H)
    center_gt = center_gt_3d[0, 0]

    # 保存结果
    gt_data = {
        "pre_path": str(pre_path).replace("\\", "/"),
        "intra_path": str(intra_path).replace("\\", "/"),
        "center_pre": center_pre[0].tolist(),
        "markers_pre": markers_pre.tolist(),
        "markers_intra": markers_intra.tolist(),
        "center_gt": center_gt.tolist(),
        "homography": H.tolist()
    }

    gt_json_path = pre_path.parent / "gt.json"
    with open(gt_json_path, "w") as f:
        json.dump(gt_data, f, indent=2)

    print(f"成功保存: {gt_json_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
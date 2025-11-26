#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coarse_search_convnext.py
- 使用 ConvNeXt 提取术中整图特征图
- 使用术前 convnext 模板向量做相似度计算生成 coarse heatmap
- 融合 y 方向解剖先验 + 二维空间先验 + 皮肤先验
- NMS 选 Top-K 作为候选，写入 candidates_convnext.json
- 不修改旧 coarse_search.py，方便对比实验
"""

import os
import json
import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    import yaml
except Exception:
    yaml = None

from convnext_utils import ConvNeXtFeatureExtractor


# ----------------- 小工具 -----------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_yaml(p, default_={}):
    if p is None or not os.path.isfile(p) or yaml is None:
        return default_
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_y_prior(H, band, smooth=True):
    """
    H: 图像高度
    band: [y_min_ratio, y_max_ratio]，如 [0.3, 0.75]
    """
    y = np.linspace(0, 1, H, dtype=np.float32)
    y0, y1 = float(band[0]), float(band[1])
    prior = np.zeros_like(y)
    mask = (y >= y0) & (y <= y1)
    prior[mask] = 1.0
    if smooth:
        # 用一个简单的余弦平滑边缘
        k = 0.05
        for i in range(H):
            if y[i] < y0 and y0 - y[i] < k:
                prior[i] = 0.5 * (1 + np.cos(np.pi * (y0 - y[i]) / k))
            if y[i] > y1 and y[i] - y1 < k:
                prior[i] = 0.5 * (1 + np.cos(np.pi * (y[i] - y1) / k))
    prior = np.clip(prior, 0, 1)
    return prior  # [H]


def make_xy_gauss_prior(H, W, cx, cy, sx, sy):
    """
    构建二维高斯先验，以 (cx,cy) 为中心，标准差 sx, sy（单位=像素）
    """
    ys = np.arange(H, dtype=np.float32)
    xs = np.arange(W, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    dx2 = (X - cx) ** 2 / (2 * sx**2 + 1e-6)
    dy2 = (Y - cy) ** 2 / (2 * sy**2 + 1e-6)
    g = np.exp(-(dx2 + dy2))
    g /= g.max() + 1e-6
    return g.astype(np.float32)


def skin_prior_ycrcb(bgr_resized):
    """
    非严格皮肤先验（简单版）：
    - 在 YCrCb 空间根据经验阈值做一个粗略皮肤掩模
    """
    img = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(img)
    # 粗略阈值，可根据数据调
    skin_mask = cv2.inRange(img, (0, 133, 77), (255, 173, 127))
    skin_mask = skin_mask.astype(np.float32) / 255.0
    # 轻度形态学操作平滑
    kernel = np.ones((5, 5), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.GaussianBlur(skin_mask, (9, 9), 0)
    if skin_mask.max() > 0:
        skin_mask /= skin_mask.max()
    return skin_mask  # [H,W], 0~1


def nms_2d(score, k=9, min_dist=20, thresh=0.2):
    """
    简单 2D NMS，在 score 上找 Top-K 局部峰值
    """
    H, W = score.shape
    score_copy = score.copy()
    coords = []
    for _ in range(k):
        idx = np.argmax(score_copy)
        v = score_copy.flat[idx]
        if v < thresh:
            break
        y, x = divmod(idx, W)
        coords.append((x, y, v))
        x0 = max(0, x - min_dist)
        x1 = min(W, x + min_dist)
        y0 = max(0, y - min_dist)
        y1 = min(H, y + min_dist)
        score_copy[y0:y1, x0:x1] = -1.0
    return coords


# ----------------- 主流程 -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intra", required=True, help="术中无贴片图像路径")
    ap.add_argument("--templates", required=True, help="旧模板目录 (含 meta.json)")
    ap.add_argument(
        "--conv_meta",
        required=True,
        help="ConvNeXt 模板 meta 路径 (build_convnext_template.py 输出)",
    )
    ap.add_argument("--out", required=True, help="候选输出 JSON 路径")
    ap.add_argument(
        "--params",
        default=None,
        help="ConvNeXt 粗搜参数 YAML (建议 configs/params_convnext.yaml)",
    )
    ap.add_argument(
        "--anatomy",
        default="auto",
        choices=["auto", "neck", "chest", "groin"],
        help="解剖区域，用于设置 y 带先验；auto 则从 conv_meta / meta.json 推断或用 chest 作为默认",
    )
    ap.add_argument(
        "--debug_dir",
        default=None,
        help="调试输出目录（保存 heatmap 等），可为空",
    )
    ap.add_argument(
        "--device",
        default=None,
        help='"cuda" / "cpu" / None(自动)',
    )
    args = ap.parse_args()

    intra_path = Path(args.intra)
    tmpl_dir = Path(args.templates)
    conv_meta_path = Path(args.conv_meta)
    out_path = Path(args.out)

    if not intra_path.is_file():
        raise FileNotFoundError(str(intra_path))
    if not (tmpl_dir / "meta.json").is_file():
        raise FileNotFoundError(f"{tmpl_dir/'meta.json'} not found.")
    if not conv_meta_path.is_file():
        raise FileNotFoundError(str(conv_meta_path))

    if args.debug_dir:
        ensure_dir(Path(args.debug_dir))

    # 1) 读取参数
    params = load_yaml(args.params, {})
    conv_cfg = params.get(
        "convnext_coarse",
        {
            "model_name": "convnext_tiny",
            "img_size": 384,
            "topk": 9,
            "min_dist": 20,
            "score_thresh": 0.2,
            "gauss_sigma_rel": [0.12, 0.20],
            "priors_weight": {"y": 0.6, "xy": 0.9, "skin": 0.8, "sim": 1.0},
        },
    )

    model_name = conv_cfg.get("model_name", "convnext_tiny")
    img_size = int(conv_cfg.get("img_size", 384))
    topk = int(conv_cfg.get("topk", 9))
    min_dist = int(conv_cfg.get("min_dist", 20))
    score_thresh = float(conv_cfg.get("score_thresh", 0.2))
    gauss_sigma_rel = conv_cfg.get("gauss_sigma_rel", [0.12, 0.20])
    priors_w = conv_cfg.get(
        "priors_weight",
        {"y": 0.6, "xy": 0.9, "skin": 0.8, "sim": 1.0},
    )

    # 解剖 y 带先验 band 设置
    meta_old = json.load(open(tmpl_dir / "meta.json", "r", encoding="utf-8"))
    conv_meta = json.load(open(conv_meta_path, "r", encoding="utf-8"))

    anatomy = args.anatomy
    if anatomy == "auto":
        # 简单从 meta 中的 anatomy 字段推断，否则默认 chest
        anatomy = meta_old.get("anatomy", "chest")
        if anatomy not in ["neck", "chest", "groin"]:
            anatomy = "chest"

    anatomy_cfg = params.get("anatomy_band", {
        "neck": [0.10, 0.45],
        "chest": [0.30, 0.75],
        "groin": [0.50, 0.95],
    })
    y_band = anatomy_cfg.get(anatomy, [0.30, 0.75])

    rel_center = conv_meta.get("rel_center_image", meta_old.get("rel_center_image", None))
    if rel_center is None:
        # 兜底：用术前中心在术前图中的相对坐标
        cx_pre = float(conv_meta["circle_center_preop"]["cx"])
        cy_pre = float(conv_meta["circle_center_preop"]["cy"])
        W_pre = float(conv_meta["preop_image_size"]["width"])
        H_pre = float(conv_meta["preop_image_size"]["height"])
        rel_center = {"rx": cx_pre / max(W_pre, 1.0), "ry": cy_pre / max(H_pre, 1.0)}

    rx = float(rel_center.get("rx", 0.5))
    ry = float(rel_center.get("ry", 0.5))

    # 2) 读取术中图像
    intra_bgr = cv2.imread(str(intra_path), cv2.IMREAD_COLOR)
    if intra_bgr is None:
        raise FileNotFoundError(str(intra_path))
    H_orig, W_orig, _ = intra_bgr.shape

    # 为了和 ConvNeXt 特征坐标对应，我们把术中图像也缩放到 img_size x img_size
    intra_resized = cv2.resize(
        intra_bgr, (img_size, img_size), interpolation=cv2.INTER_LINEAR
    )
    H_res, W_res = intra_resized.shape[:2]  # 应该等于 img_size

    # 3) ConvNeXt 特征图
    extractor = ConvNeXtFeatureExtractor(
        model_name=model_name,
        img_size=img_size,
        device=args.device,
    )
    feat_map = extractor.extract_feature_map(intra_resized)  # [C,Hf,Wf]
    C, Hf, Wf = feat_map.shape

    # 4) 相似度 heatmap
    f_tpl = np.asarray(conv_meta["template_feature"], dtype=np.float32)  # [C]
    if f_tpl.ndim != 1 or f_tpl.shape[0] != C:
        raise ValueError(
            f"模板特征维度与 ConvNeXt 特征图不匹配: {f_tpl.shape[0]} vs {C}"
        )
    # 已经 L2 归一化过，但再保险
    f_tpl /= np.linalg.norm(f_tpl) + 1e-8

    f = f_tpl.reshape(C, 1, 1)
    sim_map = (feat_map * f).sum(axis=0)  # [Hf,Wf]
    sim_map -= sim_map.min()
    if sim_map.max() > 1e-6:
        sim_map /= sim_map.max()
    sim_map = sim_map.astype(np.float32)

    # 上采样到 img_size x img_size
    sim_up = cv2.resize(
        sim_map, (W_res, H_res), interpolation=cv2.INTER_CUBIC
    )  # [H_res,W_res]

    # 5) 构建先验
    # 5.1 y 带先验
    y_prior_1d = make_y_prior(H_res, y_band, smooth=True)  # [H_res]
    y_prior = np.repeat(y_prior_1d[:, None], W_res, axis=1)  # [H_res,W_res]

    # 5.2 二维高斯先验（以术前相对坐标为中心）
    cx_p = rx * W_res
    cy_p = ry * H_res
    sx = gauss_sigma_rel[0] * W_res
    sy = gauss_sigma_rel[1] * H_res
    xy_prior = make_xy_gauss_prior(H_res, W_res, cx_p, cy_p, sx, sy)

    # 5.3 皮肤先验
    skin_p = skin_prior_ycrcb(intra_resized)

    # 6) 融合先验
    w_sim = float(priors_w.get("sim", 1.0))
    w_y = float(priors_w.get("y", 0.6))
    w_xy = float(priors_w.get("xy", 0.9))
    w_skin = float(priors_w.get("skin", 0.8))

    # 用 log 空间叠加更稳
    eps = 1e-6
    score = (
        w_sim * np.log(sim_up + eps)
        + w_y * np.log(y_prior + eps)
        + w_xy * np.log(xy_prior + eps)
        + w_skin * np.log(skin_p + eps)
    )
    score = np.exp(score)
    score -= score.min()
    if score.max() > 1e-6:
        score /= score.max()
    score_map = score.astype(np.float32)  # [H_res,W_res]

    # 7) NMS 选 Top-K
    peaks_res = nms_2d(
        score_map,
        k=topk,
        min_dist=min_dist,
        thresh=score_thresh,
    )

    # 8) 映射回原始图像坐标
    sx_scale = W_orig / float(W_res)
    sy_scale = H_orig / float(H_res)

    candidates = []
    for (x_r, y_r, s) in peaks_res:
        u = float(x_r * sx_scale)
        v = float(y_r * sy_scale)
        candidates.append(
            {
                "u": u,
                "v": v,
                "score": float(s),
                "scale": 1.0,
                "angle": 0.0,
                "source": "convnext",
            }
        )

    ensure_dir(out_path.parent)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)

    print(f"[coarse_search_convnext] candidates -> {out_path} (N={len(candidates)})")

    # 9) 简单调试可视化
    if args.debug_dir:
        dbg_dir = Path(args.debug_dir)
        ensure_dir(dbg_dir)

        # heatmap 叠加在 resized 图上
        sim_color = cv2.applyColorMap(
            (sim_up * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        overlay_sim = cv2.addWeighted(intra_resized, 0.4, sim_color, 0.6, 0)

        score_color = cv2.applyColorMap(
            (score_map * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        overlay_score = cv2.addWeighted(intra_resized, 0.4, score_color, 0.6, 0)

        # 在 overlay_score 上画出 Top-K 位置（resized 坐标）
        for (x_r, y_r, s) in peaks_res:
            cv2.drawMarker(
                overlay_score,
                (int(x_r), int(y_r)),
                (0, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=18,
                thickness=2,
            )
            cv2.putText(
                overlay_score,
                f"{s:.2f}",
                (int(x_r) + 5, int(y_r) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay_score,
                f"{s:.2f}",
                (int(x_r) + 5, int(y_r) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.imwrite(str(dbg_dir / "sim_up.png"), sim_color)
        cv2.imwrite(str(dbg_dir / "overlay_sim.png"), overlay_sim)
        cv2.imwrite(str(dbg_dir / "score_map.png"), score_color)
        cv2.imwrite(str(dbg_dir / "overlay_score_peaks.png"), overlay_score)

        # 在原图上画出映射后的候选点
        ov_orig = intra_bgr.copy()
        for (x_r, y_r, s) in peaks_res:
            u = int(x_r * sx_scale)
            v = int(y_r * sy_scale)
            cv2.drawMarker(
                ov_orig,
                (u, v),
                (0, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=18,
                thickness=2,
            )
        cv2.imwrite(str(dbg_dir / "candidates_on_intra.png"), ov_orig)

        print(f"[coarse_search_convnext] debug imgs -> {dbg_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coarse_search_convnext.py
- 创新点实现：测试时数据增强 (Test-Time Augmentation, TTA)
- 结合多源先验 (Anatomy + Skin) 进行候选点采样
- 对每个候选 Patch 进行旋转/亮度微扰，取平均相似度以抑制噪声
"""

import os
import json
import argparse
from pathlib import Path
import cv2
import numpy as np

try:
    import yaml
except ImportError:
    yaml = None
from tqdm import tqdm
from convnext_utils import ConvNeXtFeatureExtractor


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_yaml(p, default_={}):
    if p and os.path.isfile(p) and yaml:
        with open(p, "r", encoding="utf-8") as f: return yaml.safe_load(f)
    return default_


def get_skin_mask(bgr):
    """
    更严格的皮肤检测
    """
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    # 严格阈值，过滤黄色地面和背景
    # Cr: 140-180 (偏红), Cb: 100-135 (偏蓝限制)
    lower = np.array([0, 140, 100])
    upper = np.array([255, 180, 135])
    mask = cv2.inRange(ycrcb, lower, upper)

    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def augment_patch(patch):
    """
    TTA 生成器：生成 5 个版本的 Patch
    1. 原图
    2. 旋转 +5度
    3. 旋转 -5度
    4. 亮度 +10%
    5. 亮度 -10%
    """
    h, w = patch.shape[:2]
    center = (w // 2, h // 2)
    aug_list = [patch]  # 原图

    # 旋转增强
    for ang in [5, -5]:
        M = cv2.getRotationMatrix2D(center, ang, 1.0)
        rot = cv2.warpAffine(patch, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        aug_list.append(rot)

    # 亮度增强
    patch_f = patch.astype(np.float32)
    for gain in [1.1, 0.9]:
        bright = np.clip(patch_f * gain, 0, 255).astype(np.uint8)
        aug_list.append(bright)

    return aug_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intra", required=True, help="术中无贴片图像")
    ap.add_argument("--templates", required=True, help="模板目录")
    ap.add_argument("--conv_meta", required=True, help="ConvNeXt 模板 Meta")
    ap.add_argument("--out", required=True, help="输出 candidates.json")
    ap.add_argument("--params", default=None, help="参数文件")
    ap.add_argument("--anatomy", default="auto", help="解剖部位")
    ap.add_argument("--debug_dir", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    # 1. 加载配置和数据
    params = load_yaml(args.params, {})
    conv_cfg = params.get("convnext_coarse", {})

    # 读取 Meta
    conv_meta = json.load(open(args.conv_meta, "r", encoding="utf-8"))
    L_pixel = conv_meta.get("patch_context_size", 192)
    tpl_vec = np.array(conv_meta["template_feature"], dtype=np.float32)

    # 读取术中图
    intra_bgr = cv2.imread(args.intra)
    if intra_bgr is None: raise FileNotFoundError(args.intra)
    H, W = intra_bgr.shape[:2]

    if args.debug_dir: ensure_dir(Path(args.debug_dir))

    # 2. 构建先验概率图 (Prior Map)
    # 2.1 肤色先验
    skin_mask = get_skin_mask(intra_bgr)
    skin_float = skin_mask.astype(np.float32) / 255.0

    # 2.2 解剖 Y 轴先验
    meta_old = json.load(open(Path(args.templates) / "meta.json", "r", encoding="utf-8"))
    anatomy = args.anatomy if args.anatomy != "auto" else meta_old.get("anatomy", "chest")

    bands = params.get("anatomy_band", {"neck": [0.1, 0.45], "chest": [0.3, 0.75], "groin": [0.5, 0.95]})
    y_band = bands.get(anatomy, [0.3, 0.75])

    y_indices = np.arange(H)
    y_center = (y_band[0] + y_band[1]) / 2.0 * H
    y_width = (y_band[1] - y_band[0]) * H
    sigma = y_width * 0.6
    y_prior_vec = np.exp(-0.5 * ((y_indices - y_center) / sigma) ** 2)
    y_prior_map = np.tile(y_prior_vec[:, None], (1, W))

    # 2.3 综合先验
    combined_prior = y_prior_map * (skin_float + 0.05)
    combined_prior = cv2.GaussianBlur(combined_prior, (31, 31), 0)  # 平滑
    combined_prior /= (combined_prior.max() + 1e-6)

    # 3. 采样候选点 (Grid Sampling)
    prior_thresh = 0.15
    step = 16
    candidates_xy = []

    # Pad 原图以便在边缘裁切 Patch
    pad = L_pixel // 2
    img_padded = cv2.copyMakeBorder(intra_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT)

    for y in range(0, H, step):
        for x in range(0, W, step):
            if combined_prior[y, x] > prior_thresh:
                candidates_xy.append((x, y))

    print(f"[Coarse TTA] Sampling: {len(candidates_xy)} points based on prior.")

    if len(candidates_xy) == 0:
        print("[Coarse TTA] WARN: No candidates! Fallback to center.")
        cx, cy = W // 2, H // 2
        candidates_xy.append((cx, cy))

    # 4. 批量 TTA 特征提取
    extractor = ConvNeXtFeatureExtractor(model_name=conv_cfg.get("model_name", "convnext_tiny"), device=args.device)

    sim_scores = []
    batch_size_points = 8  # 每次处理8个点 -> 8 * 5 = 40 张图

    for i in tqdm(range(0, len(candidates_xy), batch_size_points), desc="ConvNeXt TTA"):
        batch_coords = candidates_xy[i: i + batch_size_points]

        all_aug_imgs = []
        counts = []

        for (cx, cy) in batch_coords:
            # 坐标变换到 Padded 图
            x0 = cx
            y0 = cy
            patch = img_padded[y0: y0 + L_pixel, x0: x0 + L_pixel]

            # TTA: 生成 5 个变体
            augs = augment_patch(patch)
            all_aug_imgs.extend(augs)
            counts.append(len(augs))

        # 批量提特征
        feats = extractor.extract_batch_features(all_aug_imgs)  # [N_total, C]

        # 计算与模板的相似度
        sims = np.dot(feats, tpl_vec)  # [N_total]

        cursor = 0
        for cnt in counts:
            # 取平均值作为最终得分 (Mean Aggregation)
            point_sims = sims[cursor: cursor + cnt]
            avg_sim = np.mean(point_sims)
            sim_scores.append(avg_sim)
            cursor += cnt

    # 5. 融合分数与 NMS
    final_candidates = []
    w_sim = 0.7
    w_prior = 0.3

    sparse_sim_map = np.zeros((H, W), dtype=np.float32)

    for idx, (cx, cy) in enumerate(candidates_xy):
        s_sim = sim_scores[idx]
        s_prior = combined_prior[cy, cx]

        # 相似度截断归一化
        s_sim_norm = max(0, s_sim)
        score = w_sim * s_sim_norm + w_prior * s_prior

        sparse_sim_map[cy, cx] = s_sim_norm

        final_candidates.append({
            "u": float(cx),
            "v": float(cy),
            "score": float(score),
            "raw_sim": float(s_sim),
            "source": "convnext_tta"
        })

    # 排序
    final_candidates.sort(key=lambda x: x["score"], reverse=True)

    # NMS
    kept = []
    min_dist = 40  # 像素
    for c in final_candidates:
        is_far = True
        for k in kept:
            dist = np.hypot(c["u"] - k["u"], c["v"] - k["v"])
            if dist < min_dist:
                is_far = False
                break
        if is_far:
            kept.append(c)
            if len(kept) >= conv_cfg.get("topk", 9):
                break

    # 6. 保存输出
    ensure_dir(Path(args.out).parent)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)

    print(f"[Coarse] Saved {len(kept)} candidates to {args.out}")

    # 7. 调试可视化
    if args.debug_dir:
        # 插值生成平滑 heatmap
        sim_vis = cv2.GaussianBlur(sparse_sim_map, (31, 31), 10)
        sim_vis = sim_vis / (sim_vis.max() + 1e-9) * 255
        sim_vis = cv2.applyColorMap(sim_vis.astype(np.uint8), cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(intra_bgr, 0.6, sim_vis, 0.4, 0)

        for i, c in enumerate(kept):
            cv2.putText(overlay, f"{i + 1}", (int(c['u']), int(c['v'])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imwrite(str(Path(args.debug_dir) / "sim_up.png"), sim_vis)
        cv2.imwrite(str(Path(args.debug_dir) / "overlay_score_peaks.png"), overlay)
        cv2.imwrite(str(Path(args.debug_dir) / "candidates_on_intra.png"), overlay)


if __name__ == "__main__":
    main()
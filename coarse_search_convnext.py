#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coarse_search_convnext.py (v4 - Prior Guided & Patch Based)
- 逻辑变更：先验筛选 -> 局部Patch裁剪 -> 批量特征比对 -> 重排序
- 解决全图特征图分辨率低、特征坍塌、背景误判的问题
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
    """更严格的皮肤检测 (针对体模/人体)"""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    # 严格阈值，过滤黄色地面和背景
    # Cr: 140-173 (偏红分量), Cb: 100-127 (偏蓝分量控制)
    lower = np.array([0, 140, 100])
    upper = np.array([255, 180, 135])
    mask = cv2.inRange(ycrcb, lower, upper)

    # 形态学操作：闭运算填孔，开运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intra", required=True)
    ap.add_argument("--templates", required=True)
    ap.add_argument("--conv_meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--params", default=None)
    ap.add_argument("--anatomy", default="auto")
    ap.add_argument("--debug_dir", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    # 1. 加载配置和数据
    params = load_yaml(args.params, {})
    conv_cfg = params.get("convnext_coarse", {})

    # 读取 Meta
    conv_meta = json.load(open(args.conv_meta, "r"))
    # 获取模板制作时的 patch 大小，如果没存则默认 192
    L_pixel = conv_meta.get("patch_context_size", 192)
    tpl_vec = np.array(conv_meta["template_feature"], dtype=np.float32)

    # 读取术中图
    intra_bgr = cv2.imread(args.intra)
    if intra_bgr is None: raise FileNotFoundError(args.intra)
    H, W = intra_bgr.shape[:2]

    if args.debug_dir: ensure_dir(Path(args.debug_dir))

    # 2. 构建先验概率图 (Prior Map)
    # 2.1 肤色先验 (强约束)
    skin_mask = get_skin_mask(intra_bgr)
    skin_float = skin_mask.astype(np.float32) / 255.0

    # 2.2 解剖位置 Y 轴先验
    # 读取 anatomy 设置
    meta_old = json.load(open(Path(args.templates) / "meta.json"))
    anatomy = args.anatomy if args.anatomy != "auto" else meta_old.get("anatomy", "chest")

    # 默认 Chest: 0.3-0.75
    bands = params.get("anatomy_band", {"neck": [0.1, 0.45], "chest": [0.3, 0.75], "groin": [0.5, 0.95]})
    y_band = bands.get(anatomy, [0.3, 0.75])

    y_indices = np.arange(H)
    y_center = (y_band[0] + y_band[1]) / 2.0 * H
    y_width = (y_band[1] - y_band[0]) * H
    # 高斯分布模拟
    sigma = y_width * 0.6
    y_prior_vec = np.exp(-0.5 * ((y_indices - y_center) / sigma) ** 2)
    y_prior_map = np.tile(y_prior_vec[:, None], (1, W))

    # 2.3 综合先验图
    # 逻辑：必须是皮肤 (skin>0)，且在解剖带附近分数更高
    # 加上一个底数 0.01 防止完全为 0 (虽然为了效率我们会过滤掉低的)
    combined_prior = y_prior_map * (skin_float + 0.05)
    combined_prior = cv2.GaussianBlur(combined_prior, (31, 31), 0)  # 平滑一下
    combined_prior /= (combined_prior.max() + 1e-6)

    if args.debug_dir:
        cv2.imwrite(str(Path(args.debug_dir) / "debug_skin.png"), skin_mask)
        cv2.imwrite(str(Path(args.debug_dir) / "debug_prior.png"), (combined_prior * 255).astype(np.uint8))

    # 3. 基于先验采样候选点 (Grid Sampling)
    # 策略：在 Prior > Threshold 的区域，每隔 Step 采一个点
    prior_thresh = 0.15  # 忽略低概率区域 (如地面)
    step = 16  # 采样步长 (像素)

    candidates_xy = []

    # Pad 原图以便在边缘裁切 Patch
    pad = L_pixel // 2
    img_padded = cv2.copyMakeBorder(intra_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT)

    for y in range(0, H, step):
        for x in range(0, W, step):
            if combined_prior[y, x] > prior_thresh:
                candidates_xy.append((x, y))

    print(f"[Coarse] Sampling: {len(candidates_xy)} points from prior map (step={step}).")

    if len(candidates_xy) == 0:
        print("[Coarse] WARN: No candidates passed prior check! Fallback to center grid.")
        # 兜底：在图像中心区域强行采一些点
        cx, cy = W // 2, H // 2
        for y in range(cy - 100, cy + 100, 40):
            for x in range(cx - 100, cx + 100, 40):
                candidates_xy.append((x, y))

    # 4. 批量提取特征 & 计算相似度
    # 准备 Patch Batch
    patch_list = []
    batch_size = 64
    extractor = ConvNeXtFeatureExtractor(model_name=conv_cfg.get("model_name", "convnext_tiny"), device=args.device)

    sim_scores = []

    # 分批处理
    for i in tqdm(range(0, len(candidates_xy), batch_size), desc="ConvNeXt Scanning"):
        batch_coords = candidates_xy[i: i + batch_size]
        batch_imgs = []

        for (cx, cy) in batch_coords:
            # 坐标变换到 Padded 图
            x0 = cx
            y0 = cy
            # 裁切 L_pixel x L_pixel
            p = img_padded[y0: y0 + L_pixel, x0: x0 + L_pixel]
            batch_imgs.append(p)

        # 提特征 [B, C]
        feats = extractor.extract_batch_features(batch_imgs)
        # 算相似度 (Dot product, 因为已 L2 norm)
        sims = np.dot(feats, tpl_vec)
        sim_scores.extend(sims.tolist())

    # 5. 融合分数 & 生成 Heatmap 可视化
    # 为了生成 sim_up.png，我们把离散点填回去
    sparse_sim_map = np.zeros((H, W), dtype=np.float32)
    final_candidates = []

    # 权重配置
    w_sim = 0.7
    w_prior = 0.3

    for idx, (cx, cy) in enumerate(candidates_xy):
        s_sim = sim_scores[idx]
        s_prior = combined_prior[cy, cx]

        # 融合分数
        # 注意：ConvNeXt 相似度可能在 [-1, 1]，通常在 [0.3, 0.9] 之间
        # 将其归一化到 [0, 1] 区间便于融合 (假设 min sim ~ 0.2)
        s_sim_norm = max(0, s_sim)

        score = w_sim * s_sim_norm + w_prior * s_prior

        sparse_sim_map[cy, cx] = s_sim_norm  # 仅用于绘图

        final_candidates.append({
            "u": float(cx),
            "v": float(cy),
            "score": float(score),
            "raw_sim": float(s_sim),
            "source": "convnext_grid"
        })

    # 6. NMS (Non-Maximum Suppression)
    final_candidates.sort(key=lambda x: x["score"], reverse=True)

    kept = []
    min_dist = 40  # 像素
    for c in final_candidates:
        # 简单空间抑制
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

    # 7. 输出结果
    ensure_dir(Path(args.out).parent)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2)

    print(f"[Coarse] Saved {len(kept)} candidates to {args.out}")
    if len(kept) > 0:
        print(
            f"  Top-1 Score: {kept[0]['score']:.4f} (Sim: {kept[0].get('raw_sim', 0):.4f}) at ({kept[0]['u']:.1f}, {kept[0]['v']:.1f})")

    # 8. 调试可视化
    if args.debug_dir:
        # 插值生成平滑 heatmap
        sim_vis = cv2.GaussianBlur(sparse_sim_map, (31, 31), 10)
        sim_vis = sim_vis / (sim_vis.max() + 1e-9) * 255
        sim_vis = cv2.applyColorMap(sim_vis.astype(np.uint8), cv2.COLORMAP_JET)

        # 叠加原图
        overlay = cv2.addWeighted(intra_bgr, 0.6, sim_vis, 0.4, 0)

        # 画 Top-K
        for i, c in enumerate(kept):
            uv = (int(c['u']), int(c['v']))
            cv2.drawMarker(overlay, uv, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(overlay, f"{i + 1}", (uv[0] + 10, uv[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imwrite(str(Path(args.debug_dir) / "sim_up.png"), sim_vis)
        cv2.imwrite(str(Path(args.debug_dir) / "overlay_score_peaks.png"), overlay)
        cv2.imwrite(str(Path(args.debug_dir) / "candidates_on_intra.png"), overlay)  # 复用一张


if __name__ == "__main__":
    main()
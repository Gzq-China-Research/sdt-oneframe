#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fine_match_convnext.py (v5 - Cycle Consistency & Fallback)
- 创新点实现：双向几何一致性校验 (Bi-directional Cycle Consistency)
- 只有通过正反双向投影校验的匹配才被认为是可靠的
- 包含完整的 ROI 提取、特征匹配和 Fallback 逻辑
"""

import os
import json
import glob
import argparse
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

try:
    import yaml
except ImportError:
    yaml = None


# --- 工具函数 ---
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def load_yaml(p, default_={}):
    if p and os.path.isfile(p) and yaml:
        with open(p, "r", encoding="utf-8") as f: return yaml.safe_load(f)
    return default_


def clamp(v, a, b):
    return max(a, min(b, v))


def to_keypoints(kp_np):
    # kp_np: [x, y, size, angle, response]
    kps = []
    for row in kp_np:
        k = cv2.KeyPoint(x=float(row[0]), y=float(row[1]), _size=float(row[2]),
                         angle=float(row[3]), response=float(row[4]))
        kps.append(k)
    return kps


def compute_homography(pts_src, pts_dst, thresh=4.0):
    if len(pts_src) < 4: return None, None
    # 优先使用 MAGSAC，没有则 RANSAC
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    H, mask = cv2.findHomography(pts_src, pts_dst, method, thresh, maxIters=2000)
    return H, mask


def check_cycle_consistency(pts_t, pts_q, H_t2q):
    """
    创新点：计算双向循环投影误差
    err = || T - (H_inv * H * T) ||
    """
    try:
        H_inv = np.linalg.inv(H_t2q)
    except np.linalg.LinAlgError:
        return float('inf')

    # 正向：T -> Q (虽然已经有 pts_q，但我们验证几何变换的一致性)
    # 这里我们验证：把 Q 投回 T，和原始 T 的距离

    ones = np.ones((len(pts_q), 1))
    pts_q_homo = np.hstack([pts_q, ones])

    # 反向投影：Q -> T'
    pts_t_back = (H_inv @ pts_q_homo.T).T

    # 归一化齐次坐标
    denom = pts_t_back[:, 2:3] + 1e-7
    pts_t_back = pts_t_back[:, :2] / denom

    # 误差：|| T - T' ||
    err = np.linalg.norm(pts_t - pts_t_back, axis=1).mean()
    return float(err)


def match_and_score(intra_gray, cand, templates_dir, meta, params, debug_dir=None):
    algo = params["template"]["features"]["algo"]
    fparams = params["template"]["features"]

    # 1. ROI 裁剪
    H_img, W_img = intra_gray.shape
    u_coarse, v_coarse = float(cand["u"]), float(cand["v"])
    radius_px = float(meta.get("circle_radius_px", 80))

    # 搜索窗口大小
    win = int(max(3.2 * radius_px, 64))

    x0 = int(max(0, u_coarse - win // 2))
    y0 = int(max(0, v_coarse - win // 2))
    x1 = int(min(W_img, x0 + win))
    y1 = int(min(H_img, y0 + win))

    roi = intra_gray[y0:y1, x0:x1]
    if roi.size < 100: return None

    # 上采样以增强弱纹理
    up = 1.5
    roi_up = cv2.resize(roi, (0, 0), fx=up, fy=up, interpolation=cv2.INTER_CUBIC)

    # 2. 提取特征
    if algo == "SIFT":
        det = cv2.SIFT_create(**fparams.get("sift", {}))
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    else:
        det = cv2.ORB_create(**fparams.get("orb", {}))
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    kq, dq = det.detectAndCompute(roi_up, None)
    if dq is None or len(kq) < 4: return None
    if algo == "SIFT": dq = dq.astype(np.float32)

    # 3. 遍历模板匹配
    best = None
    tdirs = sorted(glob.glob(str(Path(templates_dir) / "T*")))

    for tdir in tdirs:
        # 加载模板特征
        kp_path = Path(tdir) / "kp.npy"
        desc_path = Path(tdir) / "desc.npy"
        if not kp_path.exists(): continue

        kp_np = np.load(str(kp_path))
        desc = np.load(str(desc_path))

        if desc is None or len(kp_np) < 4: continue
        if algo == "SIFT": desc = desc.astype(np.float32)

        # KNN 匹配
        try:
            matches = matcher.knnMatch(desc, dq, k=2)
        except:
            continue

        good = []
        for mn in matches:
            if len(mn) == 2 and mn[0].distance < 0.8 * mn[1].distance:
                good.append(mn[0])

        if len(good) < 4: continue

        # 准备点对
        pts_t = np.float32([kp_np[m.queryIdx][:2] for m in good])
        pts_q = np.float32([kq[m.trainIdx].pt for m in good])

        # 计算单应性
        H, mask = compute_homography(pts_t, pts_q)
        if H is None: continue

        inliers = mask.ravel().astype(bool)
        n_inliers = inliers.sum()
        if n_inliers < 4: continue

        # 4. 创新点：双向一致性校验 (Cycle Consistency)
        # 仅使用内点进行校验
        pts_t_in = pts_t[inliers]
        pts_q_in = pts_q[inliers]

        cycle_err = check_cycle_consistency(pts_t_in, pts_q_in, H)

        # 阈值判定：如果几何不可逆（误差 > 2.5px），则视为不可靠匹配
        if cycle_err > 2.5:
            continue

        # 5. 映射中心点
        center_json = json.load(open(Path(tdir) / "center_px.json"))
        c_tpl = np.array([[[center_json["u"], center_json["v"]]]], dtype=np.float32)
        c_proj = cv2.perspectiveTransform(c_tpl, H)[0, 0]

        # 映射回全图坐标
        c_final_x = c_proj[0] / up + x0
        c_final_y = c_proj[1] / up + y0

        # 评分: 内点率高且循环误差低为佳
        score = (n_inliers / len(good)) - 0.1 * cycle_err

        if best is None or score > best["score"]:
            best = {
                "center_img": [float(c_final_x), float(c_final_y)],
                "score": float(score),
                "cycle_err": float(cycle_err),
                "n_inliers": int(n_inliers),
                "tdir": Path(tdir).name
            }

    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intra", required=True)
    ap.add_argument("--templates", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vis", default=None)
    ap.add_argument("--anatomy", default="chest")
    ap.add_argument("--params", default=None)
    ap.add_argument("--debug_dir", default=None)
    args = ap.parse_args()

    params = load_yaml(args.params, {})
    img = cv2.imread(args.intra, 0)  # 灰度图
    if img is None: raise FileNotFoundError(args.intra)

    meta = json.load(open(Path(args.templates) / "meta.json"))
    cands = json.load(open(args.candidates))

    if args.debug_dir: ensure_dir(args.debug_dir)

    best_overall = None

    # 遍历粗搜候选进行细配
    for i, cand in enumerate(tqdm(cands, desc="Fine Match")):
        ddir = str(Path(args.debug_dir) / f"cand_{i}") if args.debug_dir else None
        if ddir: ensure_dir(ddir)

        res = match_and_score(img, cand, args.templates, meta, params, debug_dir=ddir)

        if res:
            if best_overall is None or res["score"] > best_overall["score"]:
                best_overall = res
                best_overall["candidate_id"] = i

    # Fallback 逻辑
    is_fallback = False
    if best_overall is None:
        print("[Fine] All refined matches failed cycle consistency check. Fallback to best coarse.")
        if len(cands) > 0:
            best_coarse = cands[0]  # 默认第一个是分最高的
            best_overall = {
                "center_img": [best_coarse["u"], best_coarse["v"]],
                "score": best_coarse["score"],
                "fallback": True
            }
            is_fallback = True
        else:
            raise RuntimeError("No candidates available for fallback.")

    # 保存结果
    ensure_dir(Path(args.out).parent)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(best_overall, f, indent=2)

    # 可视化
    if args.vis:
        vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cx, cy = best_overall["center_img"]
        color = (0, 0, 255) if is_fallback else (0, 255, 0)
        cv2.drawMarker(vis_img, (int(cx), int(cy)), color, cv2.MARKER_CROSS, 30, 3)
        cv2.putText(vis_img, f"Fallback: {is_fallback}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imwrite(args.vis, vis_img)

    print(f"Fine match done. Fallback={is_fallback}, Score={best_overall['score']:.4f}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fine_match_convnext.py  (for ConvNeXt coarse candidates)
- 基本逻辑与 fine_match.py 相同：在 coarse center 附近做局部特征匹配 + 几何细配
- 区别：
  1) 对特征匹配更宽容（最低匹配点数更少）
  2) 如果所有候选都无法稳定细配，则回退到“最佳 coarse 候选”，不抛异常
     -> 确保整个 ConvNeXt pipeline 始终有输出结果
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
except Exception:
    yaml = None


# ------------------------- 通用小工具 -------------------------

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def clamp(v, a, b):
    return max(a, min(b, v))


def load_yaml(p, default_={}):
    if p is None or not os.path.isfile(p):
        return default_
    if yaml is None:
        return default_
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_keypoints(kp_np):
    """
    kp.npy: 每行 [x, y, size, angle, response]
    兼容不同版本的 OpenCV KeyPoint 构造方式
    """
    kps = []
    arr = np.asarray(kp_np)
    if arr.size == 0:
        return kps
    for row in arr:
        x = float(row[0])
        y = float(row[1])
        s = float(row[2] if len(row) > 2 else 5.0)
        ang = float(row[3] if len(row) > 3 else -1.0)
        resp = float(row[4] if len(row) > 4 else 0.0)
        try:
            k = cv2.KeyPoint(float(x), float(y), float(s), float(ang), float(resp))
        except TypeError:
            # 某些 OpenCV 版本是 _size
            k = cv2.KeyPoint(
                x=float(x),
                y=float(y),
                _size=float(s),
                angle=float(ang),
                response=float(resp),
            )
        kps.append(k)
    return kps


def make_matcher(algo="SIFT"):
    algo = (algo or "SIFT").upper()
    if algo == "SIFT":
        matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),
            dict(checks=64),
        )
        return matcher, "SIFT"
    elif algo == "ORB":
        matcher = cv2.FlannBasedMatcher(
            dict(
                algorithm=6,
                table_number=6,
                key_size=12,
                multi_probe_level=1,
            ),
            dict(checks=64),
        )
        return matcher, "ORB"
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        return matcher, "BF"


def compute_features(img, algo="SIFT", params=None, mask=None):
    algo = (algo or "SIFT").upper()
    if params is None:
        params = {}
    if algo == "SIFT":
        p = params.get(
            "sift",
            {"nfeatures": 3000, "contrastThreshold": 0.01, "edgeThreshold": 8},
        )
        det = cv2.SIFT_create(**p)
        k, d = det.detectAndCompute(img, mask)
        if d is not None and d.dtype != np.float32:
            d = d.astype(np.float32)
        return k, d, "SIFT"
    else:
        p = params.get(
            "orb",
            {"nfeatures": 3000, "scaleFactor": 1.2, "nlevels": 8},
        )
        det = cv2.ORB_create(**p)
        k = det.detect(img, mask)
        k, d = det.compute(img, k)
        if d is not None and d.dtype != np.uint8:
            d = d.astype(np.uint8)
        return k, d, "ORB"


def lowe_filter(knn, ratio=0.8):
    ms = []
    for mn in knn:
        if len(mn) < 2:
            continue
        m, n = mn[0], mn[1]
        if m.distance < ratio * n.distance:
            ms.append(m)
    return ms


def ransac_homography(xy_t, xy_q, method="USAC_MAGSAC", thresh_px=4.0):
    if len(xy_t) < 4:
        return None, None
    method_map = {
        "RANSAC": cv2.RANSAC,
        "LMEDS": cv2.LMEDS,
        "USAC_MAGSAC": getattr(cv2, "USAC_MAGSAC", cv2.RANSAC),
        "USAC_ACCURATE": getattr(cv2, "USAC_ACCURATE", cv2.RANSAC),
        "USAC_FAST": getattr(cv2, "USAC_FAST", cv2.RANSAC),
    }
    mflag = method_map.get(method, cv2.RANSAC)
    H, mask = cv2.findHomography(
        xy_t,
        xy_q,
        mflag,
        ransacReprojThreshold=float(thresh_px),
        maxIters=2000,
        confidence=0.995,
    )
    if H is None and mflag != cv2.RANSAC:
        H, mask = cv2.findHomography(
            xy_t, xy_q, cv2.RANSAC, ransacReprojThreshold=float(thresh_px)
        )
    return H, mask


def reproj_rmse(H, xy_t, xy_q, inlier_mask):
    if H is None or inlier_mask is None:
        return 1e9
    in_idx = inlier_mask.ravel().astype(bool)
    if in_idx.sum() < 4:
        return 1e9
    ones = np.ones((xy_t.shape[0], 1), np.float32)
    p = np.concatenate([xy_t, ones], 1)
    p2 = (H @ p.T).T
    p2 = p2[:, :2] / np.clip(p2[:, 2:3], 1e-6, None)
    err = np.linalg.norm(p2[in_idx] - xy_q[in_idx], axis=1)
    return float(np.sqrt(np.mean(err**2)))


def combine_score(inlier_ratio, n_inlier, rmse, weights,
                  coarse_score=0.0, center_r=None):
    """
    总评分：
    - inlier_ratio 越大越好
    - rmse 越小越好
    - center_r（最终中心相对 coarse 的位移，单位=贴片半径）越小越好
    - coarse_score 越大越好
    """
    s = 0.0
    s += float(weights.get("inlier", 1.1)) * float(inlier_ratio)
    s += float(weights.get("rmse", -0.15)) * float(-rmse)

    if center_r is not None:
        s += float(weights.get("center", -0.6)) * float(center_r)

    s += float(weights.get("coarse", 0.4)) * float(coarse_score)
    return s


# --------------------------- 对单个 coarse 候选做细配 ---------------------------

def match_and_score(intra_gray, cand, templates_dir, meta, params,
                    debug_dir=None, allow_large_shift=False):
    """
    对单个 coarse 候选做细配：
    - 与原 fine_match.py 基本相同
    - allow_large_shift=True 时，会放宽对 center 位移的限制
    """
    algo = params["template"]["features"]["algo"]
    fparams = params["template"]["features"]
    lowe_ratio = params["relocalize"]["match"].get("lowe_ratio", 0.8)
    ransac_cfg = params["relocalize"]["match"].get(
        "ransac", {"method": "USAC_MAGSAC", "thresh_px": 4.0}
    )
    score_w = params["relocalize"].get(
        "score_weights",
        {"inlier": 1.1, "rmse": -0.15, "center": -0.6, "coarse": 0.4},
    )
    center_cfg = params["relocalize"]["match"].get(
        "center",
        {"enable": True, "max_shift_r": 1.0, "alpha": 0.35},
    )

    H_img, W_img = intra_gray.shape

    # 术前估计的贴片半径（像素）
    radius_px = float(meta.get("circle_radius_px", cand.get("win", 80) / 4.0))

    # ROI：以 coarse center 为中心，只取 ~3个半径的局部
    u_coarse, v_coarse = float(cand["u"]), float(cand["v"])
    u_i, v_i = int(round(u_coarse)), int(round(v_coarse))
    win = int(max(3.2 * radius_px, 60.0))

    x0 = clamp(u_i - win // 2, 0, W_img - 1)
    y0 = clamp(v_i - win // 2, 0, H_img - 1)
    x1 = clamp(x0 + win, 1, W_img)
    y1 = clamp(y0 + win, 1, H_img)

    roi = intra_gray[y0:y1, x0:x1].copy()
    if roi.size < 100:
        return None

    # 为了对付低像素，ROI 上采样后再提特征
    up = 1.7
    roi_up = cv2.resize(
        roi,
        (int(roi.shape[1] * up), int(roi.shape[0] * up)),
        interpolation=cv2.INTER_CUBIC,
    )

    kq, dq, algo_used = compute_features(roi_up, algo, fparams)
    # 放宽：只要 >=4 个点就尝试匹配（原版是 >=8）
    if kq is None or dq is None or len(kq) < 4:
        return None

    # 遍历模板 T*
    tdirs = sorted(glob.glob(str(Path(templates_dir) / "T*")))
    if len(tdirs) == 0:
        return None

    matcher, _ = make_matcher(algo_used)
    best = None

    for tdir in tdirs:
        img_t = cv2.imread(str(Path(tdir) / "patch.png"), cv2.IMREAD_GRAYSCALE)
        if img_t is None:
            continue
        kp_np = np.load(str(Path(tdir) / "kp.npy"))
        desc = np.load(str(Path(tdir) / "desc.npy"))
        if kp_np is None or desc is None or kp_np.size == 0:
            continue

        if algo_used == "SIFT" and desc.dtype != np.float32:
            desc = desc.astype(np.float32)
        if algo_used == "ORB" and desc.dtype != np.uint8:
            desc = desc.astype(np.uint8)

        kt = to_keypoints(kp_np)
        if len(kt) < 4:
            continue

        try:
            knn = matcher.knnMatch(desc, dq, k=2)
        except cv2.error:
            if algo_used == "SIFT":
                knn = cv2.FlannBasedMatcher(
                    dict(algorithm=1, trees=5),
                    dict(checks=64),
                ).knnMatch(desc.astype(np.float32), dq.astype(np.float32), k=2)
            else:
                knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc, dq, k=2)

        good = lowe_filter(knn, lowe_ratio)
        if len(good) < 4:  # 放宽
            continue

        pts_t = np.float32([kt[m.queryIdx].pt for m in good]).reshape(-1, 2)
        pts_q = np.float32([kq[m.trainIdx].pt for m in good]).reshape(-1, 2)

        Htq, inmask = ransac_homography(
            pts_t,
            pts_q,
            method=ransac_cfg.get("method", "USAC_MAGSAC"),
            thresh_px=float(ransac_cfg.get("thresh_px", 4.0)),
        )
        if Htq is None or inmask is None:
            continue

        n_in = int(inmask.ravel().astype(bool).sum())
        if n_in < 4:  # 再加一个基础 inlier 数量限制
            continue

        inlier_ratio = float(n_in) / max(1, len(good))
        rmse = reproj_rmse(Htq, pts_t, pts_q, inmask)

        # 模板中心 -> ROI_up -> ROI -> 全图
        center_json = json.load(
            open(Path(tdir) / "center_px.json", "r", encoding="utf-8")
        )
        c_tpl = np.array([[center_json["u"], center_json["v"]]], np.float32).reshape(
            1, 1, 2
        )
        c_roi_up = cv2.perspectiveTransform(c_tpl, Htq).reshape(2)
        c_roi = np.array([c_roi_up[0] / up, c_roi_up[1] / up], np.float32)
        c_img_refined = np.array(
            [c_roi[0] + x0, c_roi[1] + y0], np.float32
        )  # 仅几何估计

        # 几何估计相对于 coarse center 的偏移（半径单位）
        dist_ref = float(
            np.hypot(c_img_refined[0] - u_coarse, c_img_refined[1] - v_coarse)
        )
        dist_ref_r = dist_ref / max(radius_px, 1.0)

        # 如果偏移太大（> max_shift_r），判为不可信：
        center_enable = center_cfg.get("enable", True)
        max_r = float(center_cfg.get("max_shift_r", 1.0))
        if allow_large_shift:
            # 对 ConvNeXt 粗搜适当放宽，比如最多允许 2 倍半径
            max_r = max(max_r, 2.0)
        if center_enable and dist_ref_r > max_r:
            # 这里直接跳过这个模板，但仍然可能有其他模板成功
            continue

        # 最终中心 = coarse_center 与 refined_center 的加权融合
        alpha = float(center_cfg.get("alpha", 0.35))  # 0~1，越小越接近 coarse
        c_final = (1.0 - alpha) * np.array([u_coarse, v_coarse], np.float32) + alpha * c_img_refined

        dist_final = float(
            np.hypot(c_final[0] - u_coarse, c_final[1] - v_coarse)
        )
        dist_final_r = dist_final / max(radius_px, 1.0)

        coarse_score = float(cand.get("score", 0.0))

        s = combine_score(
            inlier_ratio,
            n_in,
            rmse,
            score_w,
            coarse_score=coarse_score,
            center_r=dist_final_r,
        )

        cand_res = {
            "center_img": [float(c_final[0]), float(c_final[1])],  # 用融合后的中心
            "center_img_coarse": [float(u_coarse), float(v_coarse)],
            "center_img_refined": [float(c_img_refined[0]), float(c_img_refined[1])],
            "score": float(s),
            "n_inlier": int(n_in),
            "inlier_ratio": float(inlier_ratio),
            "rmse_px": float(rmse),
            "shift_px": float(dist_final),
            "shift_r": float(dist_final_r),
            "refined_shift_r": float(dist_ref_r),
            "tdir": Path(tdir).name,
        }

        if (best is None) or (cand_res["score"] > best["score"]):
            best = cand_res

        # debug 可视化
        if debug_dir:
            ddir = Path(debug_dir) / Path(tdir).name
            ensure_dir(ddir)
            vis = cv2.drawMatches(
                img_t,
                kt,
                roi_up,
                kq,
                good[:80],
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            cv2.imwrite(str(ddir / "matches_pre_ransac.png"), vis)
            ov = cv2.cvtColor(roi_up, cv2.COLOR_GRAY2BGR)
            cv2.drawMarker(
                ov,
                (int(c_roi_up[0]), int(c_roi_up[1])),
                (0, 0, 255),
                0,
                20,
                2,
            )
            cv2.imwrite(str(ddir / "center_on_roi_up.png"), ov)

    return best


# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intra", required=True)
    ap.add_argument("--templates", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vis", default=None)
    ap.add_argument(
        "--anatomy", choices=["neck", "chest", "groin", "auto"], default="chest"
    )
    ap.add_argument("--params", default=None)
    ap.add_argument("--debug_dir", default=None)
    args = ap.parse_args()

    params = load_yaml(args.params, {})
    img = cv2.imread(args.intra, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(args.intra)

    meta = json.load(open(Path(args.templates) / "meta.json", "r", encoding="utf-8"))
    cands = json.load(open(args.candidates, "r", encoding="utf-8"))
    if not isinstance(cands, list) or len(cands) == 0:
        raise RuntimeError("No candidates to refine.")

    if args.debug_dir:
        ensure_dir(args.debug_dir)

    # 记录最佳 coarse 候选——以便 fallback
    best_coarse = max(cands, key=lambda c: float(c.get("score", 0.0)))

    best_overall = None
    for i, c in enumerate(tqdm(cands, desc="candidates")):
        ddir = str(Path(args.debug_dir) / f"cand_{i+1:02d}") if args.debug_dir else None
        if ddir:
            ensure_dir(ddir)
        # 对 ConvNeXt pipeline，允许略大位移
        res = match_and_score(
            img,
            c,
            args.templates,
            meta,
            params,
            debug_dir=ddir,
            allow_large_shift=True,
        )
        if res is None:
            continue
        if (best_overall is None) or (res["score"] > best_overall["score"]):
            best_overall = {"candidate_id": i + 1, **res}

    # 如果所有候选都细配失败，则回退到 coarse 最佳候选
    if best_overall is None:
        print("[fine_match_convnext] WARNING: no valid refined match, fallback to best coarse candidate.")
        u = float(best_coarse["u"])
        v = float(best_coarse["v"])
        best_overall = {
            "candidate_id": int(cands.index(best_coarse) + 1),
            "center_img": [u, v],
            "center_img_coarse": [u, v],
            "center_img_refined": [u, v],
            "score": float(best_coarse.get("score", 0.0)),
            "n_inlier": 0,
            "inlier_ratio": 0.0,
            "rmse_px": 0.0,
            "shift_px": 0.0,
            "shift_r": 0.0,
            "refined_shift_r": 0.0,
            "tdir": "fallback_coarse",
            "fallback": True,
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(best_overall, f, indent=2, ensure_ascii=False)

    if args.vis:
        ov = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        u = int(round(best_overall["center_img"][0]))
        v = int(round(best_overall["center_img"][1]))
        cv2.drawMarker(ov, (u, v), (0, 0, 255), 0, 26, 3)
        txt = (
            f"best@{u},{v}  s={best_overall['score']:.2f} "
            f"dr={best_overall['shift_r']:.2f}"
        )
        cv2.putText(
            ov, txt, (u + 8, v - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            ov, txt, (u + 8, v - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA
        )
        cv2.imwrite(args.vis, ov)

    print("fine_match_convnext: best ->", best_overall)


if __name__ == "__main__":
    main()

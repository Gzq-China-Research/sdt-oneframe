# viz_utils.py
# 纯 OpenCV 渲染，不依赖 matplotlib
import cv2, numpy as np, json, os
from pathlib import Path

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def put_kv(img, kv, org=(10,30), scale=0.6, color=(0,255,255)):
    vis = img.copy()
    y = org[1]
    for k,v in kv.items():
        txt = f"{k}: {v}"
        cv2.putText(vis, txt, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(vis, txt, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += int(26*scale/0.6)
    return vis

def save_json(path, obj):
    ensure_dir(Path(path).parent)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def draw_keypoints(gray, kps):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.drawKeypoints(gray, kps, vis, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

def draw_matches(img1, kps1, img2, kps2, matches, inlier_mask=None):
    if inlier_mask is None:
        inlier_mask = [1]*len(matches)
    draw_params = dict(matchesMask=inlier_mask, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return cv2.drawMatches(img1, kps1, img2, kps2, matches, None, **draw_params)

def draw_inlier_lines(win, pts_src, pts_dst, inlier_mask):
    vis = cv2.cvtColor(win, cv2.COLOR_GRAY2BGR)
    for (p,d,ok) in zip(pts_src, pts_dst, inlier_mask):
        c = (0,255,0) if ok else (0,0,255)
        p = tuple(np.int32(p)); d = tuple(np.int32(d))
        cv2.line(vis, p, d, c, 1, cv2.LINE_AA)
        cv2.circle(vis, d, 2, c, -1, cv2.LINE_AA)
    return vis

def heatmap_on_image(gray, heat, alpha=0.6):
    h = heat.copy().astype(np.float32)
    h -= h.min(); m = h.max();
    if m>1e-6: h /= m
    h = (h*255).astype(np.uint8)
    h = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    if gray.ndim==2: bg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else: bg = gray.copy()
    bg = cv2.resize(bg, (h.shape[1], h.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(h, alpha, bg, 1-alpha, 0)

def plot_series(series_list, labels=None, size=(640,320), margin=40):
    # series_list: [np.array, ...] 已做Z-score或可直接绘制
    W,H = size
    canvas = np.full((H,W,3), 255, np.uint8)
    colors = [(40,40,220),(30,180,30),(220,160,30),(170,30,180)]
    if labels is None: labels = [f"s{i}" for i in range(len(series_list))]
    # 坐标轴
    cv2.rectangle(canvas, (margin,margin), (W-margin,H-margin), (0,0,0), 1, cv2.LINE_AA)
    for idx, s in enumerate(series_list):
        if len(s)==0: continue
        s = np.array(s, np.float32)
        if np.std(s) < 1e-6: s = s*0
        s = (s - s.min()) / (s.max()-s.min()+1e-6)
        pts = []
        for i,v in enumerate(s):
            x = margin + int((W-2*margin) * i/max(1,len(s)-1))
            y = margin + int((H-2*margin) * (1.0-v))
            pts.append((x,y))
        cv2.polylines(canvas, [np.array(pts,np.int32)], False, colors[idx%len(colors)], 2, cv2.LINE_AA)
        cv2.putText(canvas, labels[idx], (pts[-1][0]-60, pts[-1][1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx%len(colors)], 1, cv2.LINE_AA)
    return canvas

def render_ncc_surface(center_uv, win_size, grid, step=1):
    # grid: 2D numpy array of NCC scores
    g = grid.copy().astype(np.float32)
    g -= g.min(); M = g.max()
    if M>1e-6: g /= M
    g = (g*255).astype(np.uint8)
    g = cv2.applyColorMap(g, cv2.COLORMAP_TURBO)
    vis = g
    # 标注中心
    u,v = win_size[0]//2, win_size[1]//2
    cv2.drawMarker(vis, (u,v), (0,0,0), cv2.MARKER_CROSS, 16, 2)
    return vis

def overlay_points_on_edge(gray, points, color=(0,255,255)):
    if gray.ndim==2: vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else: vis = gray.copy()
    for p in points:
        cv2.circle(vis, (int(p[0]), int(p[1])), 2, color, -1, cv2.LINE_AA)
    return vis

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环带 NCC：
- 使用 meta.json 中的环半径与模板环带强度谱
- 在 (u*,v*) 周围 ±W 网格搜索皮尔逊相关，取峰值并做二次曲线拟合得亚像素
"""
import numpy as np
import cv2

def _ring_profile_at(gray, cx, cy, radii, thick):
    h, w = gray.shape
    Y, X = np.ogrid[:h, :w]
    rr = np.sqrt((X-cx)**2 + (Y-cy)**2)
    vals = []
    for r0 in radii:
        mask = (rr >= r0 - thick/2.0) & (rr <= r0 + thick/2.0)
        vals.append(float(gray[mask].mean()) if np.any(mask) else 0.0)
    v = np.array(vals, dtype=np.float32)
    v = (v - v.mean()) / (v.std()+1e-6)
    return v

def _corr(a,b):
    return float(np.dot(a,b) / (len(a) + 1e-6))

def refine_center_with_rings(gray, init_uv, meta, window_px=8):
    u0,v0 = init_uv
    radii = meta['radii']
    thick = meta['ring_thickness_px']
    tpl = np.array(meta['ring_profile_template'], dtype=np.float32)

    us = np.arange(int(u0-window_px), int(u0+window_px)+1)
    vs = np.arange(int(v0-window_px), int(v0+window_px)+1)
    H, W = len(vs), len(us)
    C = np.zeros((H,W), np.float32)
    for i,v in enumerate(vs):
        for j,u in enumerate(us):
            prof = _ring_profile_at(gray, u, v, radii, thick)
            C[i,j] = _corr(prof, tpl)

    i0, j0 = np.unravel_index(np.argmax(C), C.shape)
    v_peak, u_peak = vs[i0], us[j0]

    def quad_fit_1d(y1,y2,y3):
        denom = (y1 - 2*y2 + y3)
        if abs(denom) < 1e-6: return 0.0
        delta = 0.5*(y1 - y3)/denom
        return float(np.clip(delta, -1.0, 1.0))

    du=dv=0.0
    if 0<j0<W-1: du = quad_fit_1d(C[i0,j0-1], C[i0,j0], C[i0,j0+1])
    if 0<i0<H-1: dv = quad_fit_1d(C[i0-1,j0], C[i0,j0], C[i0+1,j0])

    return (u_peak + du, v_peak + dv, float(C[i0,j0]))

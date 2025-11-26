#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_from_one.py (v3.9)
- 弱光归一化：illum_norm_bgr_to_gray()
- 贴片检测：轮廓->椭圆拟合为主，HoughCircles 为辅；解剖 y 带与中心暗度比联合评分
- 新增 anatomy='chest' (y 带约 [0.30,0.75])
- 合成模板 tilt 扩到 3.0；scale 扩到 0.7~1.7
- 生成 coarse_templates: 仅保留环带邻域边缘
"""
import os, json, argparse, glob
from pathlib import Path
import numpy as np, cv2

try:
    import yaml
except Exception:
    yaml = None

# ------------------ 默认参数（与 params.yaml 对齐） ------------------
DEF = {
    'template': {
        'features': {
            'algo': 'SIFT',
            'sift': {'nfeatures': 4000, 'contrastThreshold': 0.01, 'edgeThreshold': 8},
            'orb' : {'nfeatures': 3000, 'scaleFactor': 1.2, 'nlevels': 8}
        },
        'synth': {
            'tilt': [1.0, 1.2, 1.6, 2.0, 2.4, 2.8, 3.0],
            'rot_deg_step': 30,
            'scale': [0.7, 0.85, 1.0, 1.25, 1.5, 1.7],
            'gamma': [0.8, 1.0, 1.2],
            'contrast': [0.9, 1.0, 1.1],
            'blur_sigma': [0.0, 0.8],
            'noise_sigma': [0, 5],
        },
        'rings': {'radii_scale':[1.05,1.35,1.70,2.10], 'thickness_px':8},
        'coarse': {
            'edge_low': 35, 'edge_high': 110,
            'rot_deg_step': 20,
            'scales':[0.55,0.70,0.85,1.00,1.20,1.40,1.60,1.85,2.10],
            'max_templates': 120,
            'mask_inner_ratio': 0.92,
            'ring_keep_scale':[1.15, 2.10]
        }
    }
}

# ------------------------- 工具函数 -------------------------
def load_params(path):
    if path is None or not os.path.isfile(path): return DEF
    with open(path,'r',encoding='utf-8') as f:
        return yaml.safe_load(f) if yaml is not None else DEF

def grayworld_white_balance(bgr):
    b,g,r=cv2.split(bgr.astype(np.float32))
    mb,mg,mr=b.mean()+1e-6,g.mean()+1e-6,r.mean()+1e-6
    k=(mb+mg+mr)/3.0
    b=b*(k/mb); g=g*(k/mg); r=r*(k/mr)
    out=cv2.merge([b,g,r]); return np.clip(out,0,255).astype(np.uint8)

def illum_norm_bgr_to_gray(bgr):
    # 灰世界 + CLAHE(L) + 轻降噪 + gamma=0.9
    bgr=grayworld_white_balance(bgr)
    lab=cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L,a,b=cv2.split(lab)
    L=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(L)
    lab=cv2.merge([L,a,b])
    bgr=cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    g=cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g=cv2.fastNlMeansDenoising(g, None, 7, 7, 21)
    g=(np.power(g/255.0, 0.9)*255.0).astype(np.uint8)
    return g

def canny_edge(gray, low=50, high=120):
    return cv2.Canny(gray, low, high)

def clamp(v,a,b): return max(a,min(b,v))

def anatomy_y_band(anatomy, default_band):
    if anatomy=='neck':  return [0.10,0.45]
    if anatomy=='chest': return [0.30,0.75]   # 新增：胸前/胸骨区
    if anatomy=='groin': return [0.45,0.88]
    return default_band

def draw_detect_debug(bgr, cands, pick, band=None, path_png=None):
    vis = bgr.copy()
    h,w = vis.shape[:2]
    if band is not None:
        y0=int(band[0]*h); y1=int(band[1]*h)
        cv2.rectangle(vis,(0,y0),(w-1,y1),(64,64,64),2,cv2.LINE_AA)
    for i,c in enumerate(cands[:12]):
        color=(0,255,255)
        cv2.ellipse(vis,(int(c['cx']),int(c['cy'])),
                    (int(c['a']),int(c['b'])), float(c['theta']*180/np.pi),0,360,color,2,cv2.LINE_AA)
        cv2.putText(vis,f"{i+1}:{c['score']:.2f}",(int(c['cx'])+4,int(c['cy'])-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(vis,f"{i+1}:{c['score']:.2f}",(int(c['cx'])+4,int(c['cy'])-6),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),1,cv2.LINE_AA)
    if pick is not None:
        cv2.ellipse(vis,(int(pick['cx']),int(pick['cy'])),
                    (int(pick['a']),int(pick['b'])), float(pick['theta']*180/np.pi),0,360,(0,0,255),2,cv2.LINE_AA)
        cv2.drawMarker(vis,(int(pick['cx']),int(pick['cy'])),(0,0,255),0,20,2)
    if path_png: cv2.imwrite(path_png,vis)
    return vis

def center_dark_ratio(gray,cx,cy,a,b):
    h,w = gray.shape
    Y,X=np.ogrid[:h,:w]
    # 椭圆半径（近似）：沿长短轴的归一化距离
    rr=((X-cx)/max(1.0,a))**2 + ((Y-cy)/max(1.0,b))**2
    inner=gray[rr<=0.5**2]; outer=gray[(rr>=0.9**2)&(rr<=1.2**2)]
    if inner.size<10 or outer.size<10: return 1.0
    return float(inner.mean()/(outer.mean()+1e-6))

# ---------------------- 贴片检测（椭圆） ----------------------
def detect_patch_ellipse(gray, params, anatomy='chest',
                         y_band=(0.25,0.8), rmin=6, rmax=140,
                         dbg_dir=None):
    H,W=gray.shape
    band=anatomy_y_band(anatomy, list(y_band))
    y0,y1=int(band[0]*H), int(band[1]*H)

    # 先粗二值（自适应），再 Canny
    th=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY_INV, 21, 5)
    ed = canny_edge(gray, 60, 140)
    ed = cv2.bitwise_or(ed, th)
    ed = cv2.morphologyEx(ed, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), iterations=1)

    cnts,_=cv2.findContours(ed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cands=[]
    for c in cnts:
        if len(c)<20: continue
        (cx,cy),(ma,mi),angle = cv2.fitEllipse(c)  # ma:长轴直径, mi:短轴直径
        a,b = ma/2.0, mi/2.0
        if cy<y0 or cy>y1: continue
        if a<rmin or a>rmax or b<rmin*0.6 or b>rmax: continue
        # 圆度与长短轴比
        ratio = min(a,b)/max(a,b)  # 0~1
        peri  = cv2.arcLength(c, True)
        area  = cv2.contourArea(c)
        circ  = 4*np.pi*area/(peri*peri + 1e-6)  # 1 为完美圆
        dr    = center_dark_ratio(gray,cx,cy,a,b) # 中心 / 外圈 亮度比（越小越像贴片）
        score = 0.6*ratio + 0.4*circ - 0.35*max(0.0, dr-0.9)
        cands.append({'cx':cx,'cy':cy,'a':a,'b':b,
                      'theta':np.deg2rad(angle),
                      'ratio':ratio,'circ':circ,'dark':dr,'score':float(score)})

    # 若失败，用 Hough 圆补一把
    if len(cands)==0:
        for p2 in [26,22,18,14]:
            cir=cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
                                 minDist=max(12,int(0.5*rmin)),
                                 param1=120, param2=p2,
                                 minRadius=rmin, maxRadius=rmax)
            if cir is None: continue
            for (x,y,r) in np.round(cir[0]).astype(int):
                if y<y0 or y>y1: continue
                score=1.0   # 退化得分
                cands.append({'cx':x,'cy':y,'a':r,'b':r,'theta':0.0,
                              'ratio':1.0,'circ':1.0,'dark':1.0,'score':score})
            if len(cands)>0: break

    cands.sort(key=lambda d:d['score'], reverse=True)
    pick = cands[0] if len(cands) else {'cx':W/2,'cy':H/2,'a':20,'b':18,'theta':0.0,'score':0.0}

    if dbg_dir:
        Path(dbg_dir).mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        draw_detect_debug(bgr, cands, pick, band, str(Path(dbg_dir)/'detect_debug.png'))
        with open(Path(dbg_dir)/'detect_debug.json','w',encoding='utf-8') as f:
            json.dump({'band':band,'candidates':cands[:20],'pick':pick}, f, indent=2, ensure_ascii=False)

    return pick

# ---------------------- 视几何合成工具 ----------------------
def homography_rotate_scale(center, angle_deg, scale, size_wh):
    cx,cy=center
    M2=cv2.getRotationMatrix2D((cx,cy), angle_deg, scale)
    H=np.vstack([M2,[0,0,1]]).astype(np.float32)
    return H

def homography_tilt(size_wh, tilt=1.0):
    w,h=size_wh
    if tilt<=1.001: return np.eye(3, dtype=np.float32)
    src=np.float32([[0,0],[w,0],[w,h],[0,h]])
    dy=0.14*(tilt-1.0)*h
    dst=np.float32([[0+0.5*dy,0+dy],[w-0.5*dy,0+dy],[w-0.5*dy,h-dy],[0+0.5*dy,h-dy]])
    H=cv2.getPerspectiveTransform(src,dst).astype(np.float32)
    return H

def apply_homography(img,H,dsize):
    return cv2.warpPerspective(img,H,dsize,flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)

def adjust_gamma_contrast(gray,gamma=1.0,contrast=1.0):
    g=gray.astype(np.float32)/255.0
    if abs(gamma-1.0)>1e-3: g=np.power(np.clip(g,0.0,1.0),gamma)
    g=g*contrast; g=np.clip(g,0.0,1.0)
    return (g*255.0).astype(np.uint8)

def add_noise_blur(gray, noise_sigma=0.0, blur_sigma=0.0):
    g=gray.astype(np.float32)
    if noise_sigma>0.5:
        noise=np.random.normal(0.0, noise_sigma, size=g.shape).astype(np.float32)
        g=g+noise
    if blur_sigma>0.05:
        k=int(blur_sigma*3)*2+1
        g=cv2.GaussianBlur(g,(k,k),blur_sigma)
    g=np.clip(g,0,255).astype(np.uint8)
    return g

# ---------------------- 主流程 ----------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--preop', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--params', default=None)
    ap.add_argument('--anatomy', choices=['auto','neck','chest','groin'], default='chest')
    ap.add_argument('--search_y_band', nargs=2, type=float, default=[0.25,0.80])
    ap.add_argument('--min_radius_px', type=int, default=6)
    ap.add_argument('--max_radius_px', type=int, default=140)
    ap.add_argument('--patch_diam_mm', type=float, default=10.0)
    ap.add_argument('--patch_height_mm', type=float, default=0.39)
    ap.add_argument('--mask_inner_ratio', type=float, default=0.92)
    ap.add_argument('--debug_dir', default=None)
    args = ap.parse_args()

    params=load_params(args.params)

    bgr=cv2.imread(args.preop, cv2.IMREAD_COLOR)
    if bgr is None: raise FileNotFoundError(args.preop)
    gray=illum_norm_bgr_to_gray(bgr)
    H0,W0=gray.shape

    yband=anatomy_y_band(args.anatomy, args.search_y_band)
    print(f"[build] anatomy={args.anatomy} -> band={yband}")

    pick = detect_patch_ellipse(gray, params, anatomy=args.anatomy,
                                y_band=yband, rmin=args.min_radius_px,
                                rmax=args.max_radius_px,
                                dbg_dir=(args.debug_dir if args.debug_dir else None))

    cx,cy,a,b = float(pick['cx']), float(pick['cy']), float(pick['a']), float(pick['b'])
    radius_px = float((a+b)/2.0)  # 用等效半径驱动后续环带
    patch_size=int(max(180, int(radius_px*4.5)))
    half=patch_size//2
    x0=clamp(int(round(cx))-half,0,W0-1); y0=clamp(int(round(cy))-half,0,H0-1)
    x1=clamp(x0+patch_size,1,W0); y1=clamp(y0+patch_size,1,H0)
    crop0=gray[y0:y1, x0:x1]; crop_bgr0=bgr[y0:y1, x0:x1]

    pixels_per_mm_preop = (2.0 * radius_px)/max(1e-6, args.patch_diam_mm)

    rings=params['template']['rings']
    radii=[float(radius_px*s) for s in rings['radii_scale']]
    ring_thick=int(rings['thickness_px'])

    # 频谱指纹（hex 图案在弱光仍有能量峰）
    crop_sq=crop0.copy()
    if crop_sq.shape[0]!=crop_sq.shape[1]:
        s=min(crop_sq.shape[:2]); crop_sq=cv2.resize(crop_sq,(s,s),cv2.INTER_AREA)

    def radial_spectrum_profile(gray_square, nbins=32, rmin=6):
        g=gray_square.astype(np.float32)
        h,w=g.shape
        win=cv2.createHanningWindow((w,h),cv2.CV_32F)
        G=np.fft.fftshift(np.fft.fft2(g*win)); mag=np.abs(G)
        cy,cx=h//2,w//2; Y,X=np.ogrid[:h,:w]
        R=np.sqrt((X-cx)**2+(Y-cy)**2); rmax=R.max()
        bins=np.linspace(rmin,rmax,nbins+1); prof=np.zeros(nbins,np.float32)
        for i in range(nbins):
            mask=(R>=bins[i])&(R<bins[i+1])
            if np.any(mask): prof[i]=float(mag[mask].mean())
        if prof.std()<1e-6: return prof*0
        return (prof-prof.mean())/(prof.std()+1e-6)

    hex_fft=radial_spectrum_profile(crop_sq, nbins=32, rmin=6)

    if args.debug_dir:
        Path(args.debug_dir).mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(Path(args.debug_dir)/'preop_context.png'), crop_bgr0)

    # 模板库导出目录
    outdir=Path(args.out); outdir.mkdir(parents=True,exist_ok=True)
    coarse_dir=outdir/'coarse_templates'; coarse_dir.mkdir(parents=True,exist_ok=True)

    meta={
        'circle_center_preop':[float(cx),float(cy)],
        'ellipse_ab_theta':[float(a),float(b), float(pick['theta'])],
        'circle_radius_px':float(radius_px),
        'pixels_per_mm_preop':float(pixels_per_mm_preop),
        'patch_size':int(patch_size),
        'radii':[float(r) for r in radii],
        'ring_thickness_px':int(ring_thick),
        'ring_profile_template':[],       # 这里我们在 coarse/fine 中用梯度环带，无需存均值强度
        'hex_fft_profile':[float(v) for v in hex_fft.tolist()],
        'hex_fft_nbins':32,
        'mask_inner_ratio':float(args.mask_inner_ratio),
        'patch_diam_mm':float(args.patch_diam_mm),
        'patch_height_mm':float(args.patch_height_mm),
        'preop_image_size':[int(W0), int(H0)],
        'rel_center_image': [float(cx) / float(W0), float(cy) / float(H0)]
    }

    # ---------- 细配模板：SIFT/ORB（与原逻辑相同，略去环内特征） ----------
    synth=params['template']['synth']
    feat =params['template']['features']
    algo=feat['algo'].upper()
    det = cv2.SIFT_create(**feat['sift']) if algo=='SIFT' else cv2.ORB_create(**feat['orb'])
    isSIFT = (algo=='SIFT')

    H_crop = np.array([[1,0,-x0],[0,1,-y0],[0,0,1]], dtype=np.float32)
    rot_step=int(max(5, synth['rot_deg_step'])); rot_list=list(range(0,360,rot_step))

    tid=0
    for t in synth['tilt']:
        H_tilt = homography_tilt((patch_size,patch_size), float(t))
        crop_tilt = apply_homography(crop0, H_tilt, (patch_size,patch_size))

        for s in synth['scale']:
            for ang in rot_list:
                H_rs = homography_rotate_scale((patch_size/2, patch_size/2), ang, s, (patch_size,patch_size))
                H_pre_to_tpl = H_rs @ H_tilt @ H_crop
                cen_pre = np.array([[[cx,cy]]], np.float32)
                cen_tpl = cv2.perspectiveTransform(cen_pre, H_pre_to_tpl)[0,0]
                u_c, v_c = float(cen_tpl[0]), float(cen_tpl[1])

                # 等效缩放因子
                e_x=np.array([[[cx+1,cy]]],np.float32)
                e_y=np.array([[[cx,cy+1]]],np.float32)
                ex_tpl=cv2.perspectiveTransform(e_x, H_pre_to_tpl)[0,0]
                ey_tpl=cv2.perspectiveTransform(e_y, H_pre_to_tpl)[0,0]
                sx=float(np.hypot(*(ex_tpl-[u_c,v_c]))); sy=float(np.hypot(*(ey_tpl-[u_c,v_c])))
                scale_mean=max(1e-6,(sx+sy)/2.0)
                r_inner_tpl=float(radius_px*args.mask_inner_ratio*scale_mean)

                img_rs=apply_homography(crop_tilt, H_rs, (patch_size,patch_size))

                for g in synth['gamma']:
                    for c in synth['contrast']:
                        img_adj=adjust_gamma_contrast(img_rs, float(g), float(c))
                        for bz in synth['blur_sigma']:
                            for nz in synth['noise_sigma']:
                                img_syn=add_noise_blur(img_adj, float(nz), float(bz))

                                mask=np.ones_like(img_syn, np.uint8)*255
                                cv2.circle(mask,(int(round(u_c)),int(round(v_c))),
                                           int(max(1,r_inner_tpl)),0,-1,cv2.LINE_AA)

                                if isSIFT:
                                    kp,desc=det.detectAndCompute(img_syn, mask)
                                else:
                                    kps=det.detect(img_syn, None); kp,desc=det.compute(img_syn,kps)
                                if desc is None:
                                    desc=np.zeros((0, 128 if isSIFT else 32),
                                                  dtype=(np.float32 if isSIFT else np.uint8))
                                kp_np=np.array([[k.pt[0],k.pt[1],k.size,k.angle,k.response] for k in kp], np.float32)

                                tid+=1
                                tdir=outdir/f"T{tid:04d}"; tdir.mkdir(parents=True,exist_ok=True)
                                cv2.imwrite(str(tdir/'patch.png'), img_syn)
                                np.save(str(tdir/'kp.npy'),   kp_np)
                                np.save(str(tdir/'desc.npy'), desc)
                                with open(tdir/'center_px.json','w',encoding='utf-8') as f:
                                    json.dump({'u':u_c,'v':v_c}, f, indent=2)
                                np.save(str(tdir/'H_preop_to_tpl.npy'), H_pre_to_tpl)

    # ---------- 粗搜模板（环带边缘） ----------
    c_cfg = params['template']['coarse']
    written=0
    rot_list_c=list(range(0,360, max(5, int(c_cfg.get('rot_deg_step',20)))))
    H_crop = np.array([[1,0,-x0],[0,1,-y0],[0,0,1]], dtype=np.float32)
    for t in synth['tilt']:
        if written >= int(c_cfg.get('max_templates',120)): break
        H_tilt = homography_tilt((patch_size,patch_size), float(t))
        crop_tilt = apply_homography(crop0, H_tilt, (patch_size,patch_size))
        for s in c_cfg['scales']:
            if written >= int(c_cfg.get('max_templates',120)): break
            for ang in rot_list_c:
                if written >= int(c_cfg.get('max_templates',120)): break
                H_rs = homography_rotate_scale((patch_size/2,patch_size/2), ang, float(s), (patch_size,patch_size))
                H_pre_to_tpl = H_rs @ H_tilt @ H_crop
                cen_pre = np.array([[[cx,cy]]], np.float32)
                cen_tpl = cv2.perspectiveTransform(cen_pre, H_pre_to_tpl)[0,0]
                u_c, v_c = float(cen_tpl[0]), float(cen_tpl[1])

                e_x=np.array([[[cx+1,cy]]],np.float32)
                e_y=np.array([[[cx,cy+1]]],np.float32)
                ex_tpl=cv2.perspectiveTransform(e_x, H_pre_to_tpl)[0,0]
                ey_tpl=cv2.perspectiveTransform(e_y, H_pre_to_tpl)[0,0]
                sx=float(np.hypot(*(ex_tpl-[u_c,v_c]))); sy=float(np.hypot(*(ey_tpl-[u_c,v_c])))
                scale_mean=max(1e-6,(sx+sy)/2.0)
                r_in=float(radius_px*float(c_cfg.get('mask_inner_ratio',0.92))*scale_mean)
                r_out=float(r_in * float(c_cfg.get('ring_keep_scale',[1.15,2.10])[1]))

                img_rs = apply_homography(crop_tilt, H_rs, (patch_size,patch_size))
                ed = canny_edge(img_rs, int(c_cfg.get('edge_low',35)), int(c_cfg.get('edge_high',110)))

                mask_ring = np.zeros_like(ed, np.uint8)
                cv2.circle(mask_ring,(int(round(u_c)),int(round(v_c))), int(max(1,r_out)), 255, -1, cv2.LINE_AA)
                cv2.circle(mask_ring,(int(round(u_c)),int(round(v_c))), int(max(1,r_in)),   0, -1, cv2.LINE_AA)
                ed = cv2.bitwise_and(ed, ed, mask=mask_ring)
                ed = cv2.dilate(ed, np.ones((3,3),np.uint8), 1)

                cv2.imwrite(str(coarse_dir/f"ctilt{t}_s{s}_r{ang}.png"), ed)
                written += 1

    with open(outdir/'meta.json','w',encoding='utf-8') as f:
        json.dump(meta,f,indent=2,ensure_ascii=False)

    n_tpl=len(glob.glob(str(outdir/'T*')))
    n_coarse=len(glob.glob(str(outdir/'coarse_templates/*.png')))
    print(f"Done. Templates: {n_tpl}, coarse reps: {n_coarse}")
    if args.debug_dir:
        print("build_from_one.py: debug exported.")

if __name__ == '__main__':
    main()

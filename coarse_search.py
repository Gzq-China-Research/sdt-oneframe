#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coarse_search.py (v3.9)
- 弱光归一化 illum_norm_bgr_to_gray()
- 在边缘图上做多尺度模板匹配（模板也是边缘）
- 解剖 y 先验 + 肤色先验 + 边界抑制
- 梯度环带重排；NMS 取 Top-K
"""
import os, json, glob, argparse
from pathlib import Path
import numpy as np, cv2
try:
    import yaml
except Exception:
    yaml=None
from tqdm import tqdm

# ------------ 简易工具 ------------
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)
def heatmap_on_image(gray, score):
    g = gray if gray.ndim==2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    g3 = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    s_norm = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hm = cv2.applyColorMap(s_norm, cv2.COLORMAP_JET)
    return cv2.addWeighted(g3, 0.35, hm, 0.65, 0)

def load_params(p, DEF):
    if p is None or not os.path.isfile(p): return DEF
    with open(p,'r',encoding='utf-8') as f:
        return yaml.safe_load(f) if yaml is not None else DEF

# ----------- 成像归一化 -----------
def grayworld_white_balance(bgr):
    b,g,r=cv2.split(bgr.astype(np.float32))
    mb,mg,mr=b.mean()+1e-6,g.mean()+1e-6,r.mean()+1e-6
    k=(mb+mg+mr)/3.0
    b=b*(k/mb); g=g*(k/mg); r=r*(k/mr)
    out=cv2.merge([b,g,r]); return np.clip(out,0,255).astype(np.uint8)

def illum_norm_bgr_to_gray(bgr):
    bgr=grayworld_white_balance(bgr)
    lab=cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L,a,b=cv2.split(lab)
    L=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(L)
    lab=cv2.merge([L,a,b])
    bgr=cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    g=cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g=cv2.fastNlMeansDenoising(g, None, 7, 7, 21)
    g=(np.power(g/255.0, 0.9)*255.0).astype(np.uint8)
    return g, bgr

def make_y_prior(H, anatomy='chest', sigma=0.12, band=None):
    if band is None:
        band = [0.30,0.75] if anatomy=='chest' else ([0.10,0.45] if anatomy=='neck' else [0.45,0.88])
    y = np.linspace(0,1,H,endpoint=True).reshape(-1,1).astype(np.float32)
    ym = 0.5*(band[0]+band[1])
    w  = np.exp(-0.5*((y-ym)/(sigma+1e-6))**2)
    return w

def skin_prior_ycrcb(bgr):
    ycrcb=cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y,Cr,Cb=cv2.split(ycrcb)
    # 宽松阈值（适配弱光）
    skin=((Cr>130)&(Cr<180)&(Cb>85)&(Cb<135)).astype(np.uint8)*255
    skin=cv2.GaussianBlur(skin,(9,9),0)
    skin=skin.astype(np.float32)/255.0
    return 0.4 + 0.6*skin  # 0.4~1.0 权重

def nms_points(pts, min_dist=40):
    pts=sorted(pts, key=lambda x:x['score'], reverse=True)
    kept=[]
    for p in pts:
        ok=True
        for q in kept:
            if (p['u']-q['u'])**2 + (p['v']-q['v'])**2 < (min_dist**2):
                ok=False; break
        if ok: kept.append(p)
    return kept

def ring_profile(base,u,v,radii,thick):
    h,w=base.shape; Y,X=np.ogrid[:h,:w]
    rr=np.sqrt((X-u)**2 + (Y-v)**2)
    vals=[]
    for r0 in radii:
        mask=(rr>=r0-thick/2)&(rr<=r0+thick/2)
        vals.append(float(base[mask].mean()) if np.any(mask) else 0.0)
    v=np.array(vals,np.float32)
    if v.std()<1e-6: return v*0
    return (v-v.mean())/(v.std()+1e-6)

# ----------------- 默认参数 -----------------
DEF = {
  'relocalize':{
    'coarse':{
      'pyramid_scales':[0.5,0.7,1.0],
      'template_scales':[0.45,0.6,0.8,1.0,1.2,1.5,1.8,2.2],
      'pre_topk':900,
      'ring_weight':0.65,
      'ring_mode':'grad',           # 'grad' or 'int'
      'nms_min_dist':42,
      'topk':9,
      'edge_low':45,
      'edge_high':135,
      'y_prior':{
        'enable': True,
        'band': [0.30, 0.75],       # chest
        'sigma': 0.10
      },
      'skin_prior_weight': 0.7,
      'border_margin_px': 28
    }
  }
}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--intra', required=True)
    ap.add_argument('--templates', required=True)
    ap.add_argument('--params', default=None)
    ap.add_argument('--roi', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--debug_dir', default=None)
    ap.add_argument('--anatomy', choices=['auto','neck','chest','groin'], default='chest')
    args=ap.parse_args()

    params=load_params(args.params, DEF)
    cs=params['relocalize']['coarse']

    bgr_full=cv2.imread(args.intra, cv2.IMREAD_COLOR)
    if bgr_full is None: raise FileNotFoundError(args.intra)
    gray_full, bgr_full = illum_norm_bgr_to_gray(bgr_full)

    # ROI
    roi=None
    if args.roi and os.path.isfile(args.roi):
        tmp=json.load(open(args.roi,'r',encoding='utf-8'))
        roi={k:int(tmp[k]) for k in ['x0','y0','x1','y1']}
    if roi:
        x0,y0,x1,y1=roi['x0'],roi['y0'],roi['x1'],roi['y1']
        img=gray_full[y0:y1, x0:x1]
        bgr=bgr_full[y0:y1, x0:x1]
    else:
        x0=y0=0; x1=gray_full.shape[1]; y1=gray_full.shape[0]
        img=gray_full; bgr=bgr_full

    meta=json.load(open(Path(args.templates)/'meta.json','r',encoding='utf-8'))
    radii=meta['radii']; thick=int(meta['ring_thickness_px'])
    ctpls=sorted(glob.glob(str(Path(args.templates)/'coarse_templates/*.png')))
    if len(ctpls)==0: raise RuntimeError('No coarse templates found.')

    H0,W0=img.shape
    all_scores=np.zeros((H0,W0), np.float32)

    dbg_dir=Path(args.debug_dir) if args.debug_dir else None
    if dbg_dir: ensure_dir(dbg_dir)

    # ---------- 多金字塔 + 边缘图模板匹配 ----------
    for s in cs['pyramid_scales']:
        small=cv2.resize(img,(max(1,int(W0*s)), max(1,int(H0*s))), interpolation=cv2.INTER_AREA)
        edge_small = cv2.Canny(small, int(cs.get('edge_low',45)), int(cs.get('edge_high',135)))
        edge_small = cv2.GaussianBlur(edge_small, (3,3), 0)
        acc=np.zeros_like(edge_small, np.float32)

        for tp in tqdm(ctpls, desc=f'coarse s={s}'):
            tpl=cv2.imread(tp, cv2.IMREAD_GRAYSCALE)
            if tpl is None: continue
            for ts in cs['template_scales']:
                tw,th=max(5,int(tpl.shape[1]*ts)), max(5,int(tpl.shape[0]*ts))
                if th>edge_small.shape[0] or tw>edge_small.shape[1]: continue
                tt=cv2.resize(tpl,(tw,th), cv2.INTER_LINEAR)
                res=cv2.matchTemplate(edge_small, tt, cv2.TM_CCOEFF_NORMED).astype(np.float32)
                # 累积到 acc（边界对齐）
                Hh, Ww = res.shape
                acc[:Hh, :Ww] = np.maximum(acc[:Hh, :Ww], res)

        acc_up=cv2.resize(acc,(W0,H0), cv2.INTER_LINEAR)
        all_scores=np.maximum(all_scores, acc_up)
        if dbg_dir:
            cv2.imwrite(str(dbg_dir/f'pyr_{s:.2f}_heat.png'), heatmap_on_image(img, acc_up))

    # ---------- 先验：y + 肤色 + 边界 ----------
    prior = np.ones_like(all_scores, np.float32)
    if cs.get('y_prior',{}).get('enable', True):
        yp = make_y_prior(H0, anatomy=args.anatomy,
                          sigma=cs['y_prior'].get('sigma',0.10),
                          band=cs['y_prior'].get('band',[0.30,0.75]))
        prior *= yp
    if cs.get('skin_prior_weight',0)>0:
        sp = skin_prior_ycrcb(bgr)
        sp = cv2.resize(sp, (W0,H0), cv2.INTER_LINEAR)
        prior = (1.0 - cs['skin_prior_weight'])*prior + cs['skin_prior_weight']*(prior*sp)

    m = int(cs.get('border_margin_px',28))
    if m>0:
        prior[:, :m] *= 0.1; prior[:, -m:] *= 0.1
        prior[:m, :] *= 0.1; prior[-m:, :] *= 0.1

    all_scores *= prior
    # ---------- 新增：二维空间先验（利用术前贴片在整幅图中的相对位置） ----------
    rel = meta.get('rel_center_image', None)
    if rel is not None and len(rel) == 2:
        cx_rel, cy_rel = float(rel[0]), float(rel[1])

        # 术中整幅图的尺寸
        H_full, W_full = gray_full.shape[:2]
        cx_exp = cx_rel * W_full
        cy_exp = cy_rel * H_full

        sigma_x_rel = float(cs.get('xy_prior_sigma_x', 0.10))  # 相对宽度
        sigma_y_rel = float(cs.get('xy_prior_sigma_y', 0.08))  # 相对高度

        # 当前 ROI 内每个像素在整幅图中的坐标
        X, Y = np.meshgrid(
            np.arange(W0, dtype=np.float32) + float(x0),
            np.arange(H0, dtype=np.float32) + float(y0)
        )
        dx = (X - cx_exp) / (sigma_x_rel * W_full + 1e-6)
        dy = (Y - cy_exp) / (sigma_y_rel * H_full + 1e-6)
        prior_xy = np.exp(-0.5 * (dx * dx + dy * dy)).astype(np.float32)

        all_scores *= prior_xy

        # （可选）输出一个调试图看看二维先验长什么样
        if dbg_dir is not None:
            prior_vis = cv2.normalize(prior_xy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imwrite(str(dbg_dir / 'prior_xy_heat.png'),
                        heatmap_on_image(img, prior_vis))
    # ---------- 阶段1：密集峰 ----------
    pre_pts=[]
    res=all_scores.copy()
    r_small=max(1, cs['nms_min_dist']//4)
    for _ in range(max(1, int(cs.get('pre_topk',900)))):
        minV,maxV,minL,maxL=cv2.minMaxLoc(res)
        u,v=maxL[0], maxL[1]
        pre_pts.append({'u':float(u+x0),'v':float(v+y0),'score':float(maxV),'win':int(meta.get('patch_size',180))})
        cv2.circle(res,(u,v), r_small, 0, -1)

    if dbg_dir:
        ov=cv2.cvtColor(gray_full, cv2.COLOR_GRAY2BGR)
        if roi: cv2.rectangle(ov,(x0,y0),(x1,y1),(0,255,0),2,cv2.LINE_AA)
        for p in pre_pts[:300]:
            cv2.circle(ov,(int(p['u']),int(p['v'])),2,(0,255,255),-1,cv2.LINE_AA)
        cv2.imwrite(str(dbg_dir/'pre_peaks_overlay.png'), ov)

    # ---------- 阶段2：环带重排（梯度） ----------
    gx = cv2.Sobel(gray_full, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_full, cv2.CV_32F, 0, 1, ksize=3)
    gmag = cv2.magnitude(gx, gy)
    rp_tpl = np.array(meta.get('ring_profile_template',[]), np.float32)
    # 若术前未存环带强度，退化为“等半径高斯环”近似
    if rp_tpl.size==0:
        rp_tpl = np.ones((len(radii),), np.float32)

    for p in pre_pts:
        rp = ring_profile(gmag, p['u'], p['v'], radii, thick)
        sim = float(np.dot(rp_tpl, rp)/(np.linalg.norm(rp_tpl)*np.linalg.norm(rp)+1e-6))
        p['score'] = (1.0-cs['ring_weight'])*p['score'] + cs['ring_weight']*sim

    final_pts = nms_points(pre_pts, cs['nms_min_dist'])[:cs['topk']]

    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    with open(args.out,'w',encoding='utf-8') as f:
        json.dump(final_pts, f, indent=2, ensure_ascii=False)

    if dbg_dir:
        ov=cv2.cvtColor(gray_full, cv2.COLOR_GRAY2BGR)
        if roi: cv2.rectangle(ov,(x0,y0),(x1,y1),(0,255,0),2,cv2.LINE_AA)
        for i,p in enumerate(final_pts):
            cv2.drawMarker(ov,(int(p['u']),int(p['v'])),
                           (0,0,255) if i==0 else (0,255,255), 0, 24, 2)
            cv2.putText(ov,f"{i+1}:{p['score']:.2f}",(int(p['u'])+6,int(p['v'])-6),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),3,cv2.LINE_AA)
            cv2.putText(ov,f"{i+1}:{p['score']:.2f}",(int(p['u'])+6,int(p['v'])-6),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),1,cv2.LINE_AA)
        cv2.imwrite(str(dbg_dir/'ring_reranked_overlay.png'), ov)

    print('Candidates saved:', args.out)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/select_roi.py (v1.1)
- 单次矩形 ROI 框选；输出 JSON：{"x0":..,"y0":..,"x1":..,"y1":..,"image_shape":[H,W]}
操作：
  拖拽左键绘制矩形；S 保存；R 重置；Q/ESC 退出（有修改会提示）
"""
import os, json, argparse
from pathlib import Path
import cv2

class ROISelector:
    def __init__(self, image_path, out_path):
        self.image_path = image_path
        self.out_path = out_path
        self.img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if self.img is None:
            raise FileNotFoundError(image_path)
        self.h, self.w = self.img.shape[:2]
        self.win = 'select_roi'
        self.pt0 = None
        self.pt1 = None
        self.dragging = False
        self.dirty = False
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        cv2.setMouseCallback(self.win, self.on_mouse)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pt0 = (x,y); self.pt1 = (x,y); self.dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.pt1 = (x,y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.pt1 = (x,y); self.dirty = True

    def get_rect(self):
        if self.pt0 is None or self.pt1 is None:
            return None
        x0 = max(0, min(self.pt0[0], self.pt1[0]))
        y0 = max(0, min(self.pt0[1], self.pt1[1]))
        x1 = min(self.w, max(self.pt0[0], self.pt1[0]))
        y1 = min(self.h, max(self.pt0[1], self.pt1[1]))
        if x1-x0 < 4 or y1-y0 < 4:
            return None
        return (x0,y0,x1,y1)

    def render(self):
        vis = self.img.copy()
        r = self.get_rect()
        if r is not None:
            x0,y0,x1,y1 = r
            cv2.rectangle(vis, (x0,y0), (x1,y1), (0,255,255), 2, cv2.LINE_AA)
        y = 22
        def put(t, color=(255,255,255)):
            nonlocal y
            cv2.putText(vis, t, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(vis, t, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
            y += 22
        put(f"{Path(self.image_path).name}  size={self.w}x{self.h}")
        put("拖拽左键:画框   R:重置   S:保存   Q/ESC:退出", (0,255,255))
        if self.dirty: put("有未保存更改", (0,200,255))
        return vis

    def save(self):
        r = self.get_rect()
        if r is None:
            print("未选择有效ROI，未保存。"); return False
        x0,y0,x1,y1 = r
        out = {"x0":int(x0),"y0":int(y0),"x1":int(x1),"y1":int(y1),"image_shape":[int(self.h),int(self.w)]}
        Path(self.out_path).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(self.out_path,'w',encoding='utf-8'), indent=2)
        print("ROI saved ->", self.out_path)
        self.dirty = False
        return True

    def loop(self):
        pending_quit = False
        while True:
            cv2.imshow(self.win, self.render())
            k = cv2.waitKey(30) & 0xFF
            if k == 255: continue
            if k in (ord('q'), 27):
                if self.dirty and not pending_quit:
                    print("有未保存更改。按 S 保存；再次按 Q/ESC 放弃更改退出。")
                    pending_quit = True
                    continue
                break
            elif k == ord('r'):
                self.pt0=self.pt1=None; self.dirty=False
            elif k == ord('s'):
                self.save()
        cv2.destroyAllWindows()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    ROISelector(args.image, args.out).loop()

if __name__ == '__main__':
    main()

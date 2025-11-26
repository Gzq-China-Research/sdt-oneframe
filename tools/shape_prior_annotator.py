#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/shape_prior_annotator.py (v1.1)
- 在术前单图上标注“解剖线/散点”先验，输出 JSON：
  {
    "lines": [ [[x1,y1],[x2,y2],...], ... ],
    "points": [[xp,yp], ...],
    "coord": "preop_abs_px"
  }
操作：
  左键：加点；线模式下连成折线（右键/Enter 完成该线）
  L：线模式   P：点模式
  U：撤销当前线最后顶点
  R：删除最后一条已完成的线
  D：删除最后一个散点
  C：清空当前未完成的线
  S：保存 JSON
  H：显示/隐藏帮助
  Q/ESC：退出（未保存会提示）
"""
import os, json, argparse
from pathlib import Path
import numpy as np
import cv2

HUD = True
MODE_LINE = 0
MODE_POINT = 1

class Annotator:
    def __init__(self, img_path, out_path, load_path=None):
        self.img_path = img_path
        self.out_path = out_path
        self.img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if self.img is None:
            raise FileNotFoundError(img_path)
        self.h, self.w = self.img.shape[:2]
        self.mode = MODE_LINE
        self.lines = []
        self.points = []
        self.curr_line = []
        self.dirty = False
        if load_path and os.path.isfile(load_path):
            try:
                d = json.load(open(load_path,'r',encoding='utf-8'))
                self.lines = d.get('lines', [])
                self.points = d.get('points', [])
                print(f"Loaded prior from {load_path}  (lines={len(self.lines)}, points={len(self.points)})")
            except Exception as e:
                print('WARN: failed to load prior:', e)
        self.win = 'shape_prior_annotator'
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        cv2.setMouseCallback(self.win, self.on_mouse)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == MODE_LINE:
                self.curr_line.append([int(x), int(y)]); self.dirty = True
            else:
                self.points.append([int(x), int(y)]); self.dirty = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.finish_line()

    def finish_line(self):
        if len(self.curr_line) >= 2:
            self.lines.append(self.curr_line.copy())
            self.curr_line.clear()
            self.dirty = True
        else:
            self.curr_line.clear()

    def render(self):
        vis = self.img.copy()
        for L in self.lines:
            for i in range(len(L)-1):
                p1 = tuple(L[i]); p2 = tuple(L[i+1])
                cv2.line(vis, p1, p2, (0,255,255), 2, cv2.LINE_AA)
            for p in L:
                cv2.circle(vis, tuple(p), 3, (0,180,255), -1, cv2.LINE_AA)
        if len(self.curr_line) > 0:
            for i in range(len(self.curr_line)-1):
                cv2.line(vis, tuple(self.curr_line[i]), tuple(self.curr_line[i+1]), (0,255,0), 2, cv2.LINE_AA)
            for p in self.curr_line:
                cv2.circle(vis, tuple(p), 3, (0,255,0), -1, cv2.LINE_AA)
        for p in self.points:
            cv2.circle(vis, tuple(p), 4, (255,0,255), -1, cv2.LINE_AA)
        if HUD:
            y = 24
            def put(t, color=(255,255,255)):
                nonlocal y
                cv2.putText(vis, t, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(vis, t, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
                y += 22
            put(f"{Path(self.img_path).name}  size={self.w}x{self.h}")
            put(f"MODE: {'LINE' if self.mode==MODE_LINE else 'POINT'}  |  lines={len(self.lines)} points={len(self.points)}  |  dirty={'Y' if self.dirty else 'N'}", (0,255,255))
            put("L线 P点 Enter/右键结束线 U撤销点 R删线 D删点 C清线 S保存 H帮助 Q/ESC退出", (200,255,200))
        return vis

    def save(self):
        out = {'lines': self.lines, 'points': self.points, 'coord': 'preop_abs_px'}
        Path(self.out_path).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(self.out_path,'w',encoding='utf-8'), indent=2)
        print('Saved prior ->', self.out_path)
        self.dirty = False

    def loop(self):
        global HUD
        pending_quit = False
        while True:
            vis = self.render()
            cv2.imshow(self.win, vis)
            k = cv2.waitKey(30) & 0xFF
            if k == 255:  continue
            if k in (ord('q'), 27):
                if self.dirty and not pending_quit:
                    print('Unsaved changes. Press S to save; press Q/ESC again to quit without saving.')
                    pending_quit = True
                    continue
                break
            elif k == ord('l'):
                self.mode = MODE_LINE
            elif k == ord('p'):
                self.mode = MODE_POINT
            elif k in (13, ord('\r')):  # Enter
                self.finish_line()
            elif k == ord('u'):
                if len(self.curr_line) > 0: self.curr_line.pop(); self.dirty = True
            elif k == ord('r'):
                if len(self.lines) > 0: self.lines.pop(); self.dirty = True
            elif k == ord('d'):
                if len(self.points) > 0: self.points.pop(); self.dirty = True
            elif k == ord('c'):
                if len(self.curr_line) > 0: self.curr_line.clear(); self.dirty = True
            elif k == ord('s'):
                self.save()
            elif k == ord('h'):
                HUD = not HUD
        cv2.destroyAllWindows()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--load', default=None)
    args = ap.parse_args()
    Annotator(args.image, args.out, args.load).loop()

if __name__ == '__main__':
    main()

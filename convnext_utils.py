#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convnext_utils.py (v2)
- 提供基于 ConvNeXt 的特征提取工具类
- 兼容不同版本 torchvision：
  - 如果有新的 Weights API（ConvNeXt_Tiny_Weights 等），优先使用
  - 否则退回到 pretrained=True，并使用经典 ImageNet 均值/方差
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.models as models


class ConvNeXtFeatureExtractor:
    """
    使用 torchvision ConvNeXt 作为特征提取器：
    - extract_global_feature: 对一个 patch 输出全局模板特征向量 [C]
    - extract_feature_map: 对一幅图输出空间特征图 [C, Hf, Wf]，每个位置相当于一个 token
    """

    def __init__(
        self,
        model_name: str = "convnext_tiny",
        img_size: int = 384,
        device: str = None,
    ):
        """
        model_name: "convnext_tiny" / "convnext_small" 等
        img_size:   输入 ConvNeXt 的图像大小（宽高相同），同时作为粗搜坐标系
        device:     "cuda" / "cpu" / None（自动）
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.img_size = int(img_size)

        # ---- 加载 ConvNeXt backbone + 兼容不同版本的权重接口 ----
        backbone = None
        mean = None
        std = None

        if model_name == "convnext_tiny":
            # 优先尝试新 Weights API
            weights_enum = getattr(models, "ConvNeXt_Tiny_Weights", None)
            if weights_enum is not None:
                try:
                    weights = weights_enum.IMAGENET1K_V1
                    backbone = models.convnext_tiny(weights=weights)
                    meta = getattr(weights, "meta", {})
                    mean = meta.get("mean", None)
                    std = meta.get("std", None)
                except Exception:
                    backbone = None

            # 如果上面失败或没有 meta，就退回 pretrained=True + 经典 ImageNet mean/std
            if backbone is None:
                try:
                    backbone = models.convnext_tiny(pretrained=True)
                except TypeError:
                    # 部分版本用 weights=None 表示随机初始化，这里强制指定预训练
                    backbone = models.convnext_tiny(weights="IMAGENET1K_V1")
        elif model_name == "convnext_small":
            weights_enum = getattr(models, "ConvNeXt_Small_Weights", None)
            if weights_enum is not None:
                try:
                    weights = weights_enum.IMAGENET1K_V1
                    backbone = models.convnext_small(weights=weights)
                    meta = getattr(weights, "meta", {})
                    mean = meta.get("mean", None)
                    std = meta.get("std", None)
                except Exception:
                    backbone = None

            if backbone is None:
                try:
                    backbone = models.convnext_small(pretrained=True)
                except TypeError:
                    backbone = models.convnext_small(weights="IMAGENET1K_V1")
        else:
            raise ValueError(f"Unsupported ConvNeXt model_name: {model_name}")

        # 如果 mean/std 还没取到，就用经典 ImageNet 归一化参数
        if mean is None or std is None:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        self.backbone = backbone.features.to(self.device).eval()
        self.mean = mean
        self.std = std

        # transform: 不做 Resize（我们用 cv2 控制大小），只做 ToTensor + Normalize
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def _preprocess_bgr(self, bgr: np.ndarray) -> torch.Tensor:
        """
        输入: BGR uint8 (H, W, 3)
        输出: 预处理后的 tensor [1,3,img_size,img_size]
        """
        if bgr is None or bgr.size == 0:
            raise ValueError("Empty image given to ConvNeXtFeatureExtractor.")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # 统一缩放到 img_size x img_size
        resized = cv2.resize(
            rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR
        )
        x = self.transform(resized).unsqueeze(0).to(self.device)  # [1,3,H,W]
        return x

    def extract_global_feature(self, bgr: np.ndarray) -> np.ndarray:
        """
        对一个 patch 提取全局特征向量 [C]：
        - 先缩放到固定大小
        - 经过 ConvNeXt backbone
        - 全局平均池化得到 [C]
        - L2 归一化
        """
        x = self._preprocess_bgr(bgr)
        with torch.no_grad():
            feat = self.backbone(x)  # [1,C,Hf,Wf]
            feat = feat.mean(dim=[2, 3])[0]  # [C]
        f = feat.cpu().numpy().astype("float32")
        norm = np.linalg.norm(f) + 1e-8
        f /= norm
        return f  # [C], 已归一化

    def extract_feature_map(self, bgr: np.ndarray) -> np.ndarray:
        """
        对整幅图像提取特征图 [C,Hf,Wf]，并对每个 (x,y) 的 feature 做 L2 归一化。
        注意：图像会被缩放到 [img_size,img_size]，粗搜坐标先在该尺度上进行。
        """
        x = self._preprocess_bgr(bgr)
        with torch.no_grad():
            feat_map = self.backbone(x)[0].cpu().numpy().astype("float32")  # [C,Hf,Wf]
        C, Hf, Wf = feat_map.shape
        flat = feat_map.reshape(C, -1)
        flat /= np.linalg.norm(flat, axis=0, keepdims=True) + 1e-8
        feat_norm = flat.reshape(C, Hf, Wf)  # [C,Hf,Wf]
        return feat_norm

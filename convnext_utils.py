#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convnext_utils.py (v3 - Optimized for Batch Processing)
- 提供基于 ConvNeXt 的特征提取工具类
- 新增 extract_batch_features: 支持批量 Patch 提取，提高推理效率
- 显式加入 AdaptiveAvgPool2d，支持任意尺寸输入
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models


class ConvNeXtFeatureExtractor:
    """
    使用 torchvision ConvNeXt 作为特征提取器：
    - extract_global_feature: 单个 Patch -> [C] (兼容旧接口)
    - extract_batch_features: 批量 Patch -> [B, C] (新核心接口)
    """

    def __init__(
            self,
            model_name: str = "convnext_tiny",
            img_size: int = 224,  # 网络输入层大小
            device: str = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.img_size = int(img_size)

        # ---- 加载 ConvNeXt backbone ----
        backbone = None
        # 尝试加载预训练权重
        if model_name == "convnext_tiny":
            try:
                # 新版 torchvision 写法
                weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
                backbone = models.convnext_tiny(weights=weights)
            except AttributeError:
                # 旧版写法
                try:
                    backbone = models.convnext_tiny(weights="IMAGENET1K_V1")
                except:
                    backbone = models.convnext_tiny(pretrained=True)
        elif model_name == "convnext_small":
            try:
                weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1
                backbone = models.convnext_small(weights=weights)
            except AttributeError:
                try:
                    backbone = models.convnext_small(weights="IMAGENET1K_V1")
                except:
                    backbone = models.convnext_small(pretrained=True)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        # 提取特征层，去掉分类头
        self.backbone = backbone.features.to(self.device).eval()

        # 显式定义全局平均池化，确保任意输入尺寸都能变成 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 标准 ImageNet 归一化
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
        ])

    def _preprocess_batch(self, bgr_list):
        """将 BGR 图片列表预处理为 Tensor Batch [B, 3, H, W]"""
        batch_tensors = []
        for img in bgr_list:
            if img is None or img.size == 0:
                # 异常处理：塞一个黑图
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

            # 转 RGB 并缩放到网络输入大小 (通常 224)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            batch_tensors.append(self.transform(resized))

        if not batch_tensors:
            return None

        return torch.stack(batch_tensors).to(self.device)

    def extract_batch_features(self, bgr_list: list) -> np.ndarray:
        """
        批量提取特征
        输入: list of BGR images (任意大小)
        输出: numpy array [B, C], L2 normalized
        """
        if not bgr_list:
            return np.array([])

        x = self._preprocess_batch(bgr_list)

        with torch.no_grad():
            # Backbone 提取特征 [B, C, Hf, Wf]
            feat_map = self.backbone(x)
            # 全局池化 [B, C, 1, 1]
            feat_vec = self.avgpool(feat_map)
            # 展平 [B, C]
            feat_vec = feat_vec.flatten(1)

        f = feat_vec.cpu().numpy().astype("float32")

        # L2 归一化
        norm = np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
        f /= norm

        return f

    def extract_global_feature(self, bgr: np.ndarray) -> np.ndarray:
        """兼容旧接口：单张图片提取"""
        feats = self.extract_batch_features([bgr])
        return feats[0]

    def extract_feature_map(self, bgr: np.ndarray) -> np.ndarray:
        """
        兼容旧接口：提取空间特征图 [C, Hf, Wf]
        注意：尽量不要用这个做粗搜，因为分辨率太低且缺乏 Context
        """
        x = self._preprocess_batch([bgr])
        with torch.no_grad():
            feat_map = self.backbone(x)[0]  # [C, Hf, Wf]

        f = feat_map.cpu().numpy().astype("float32")
        # 对每个空间位置做 L2 归一化
        C, Hf, Wf = f.shape
        flat = f.reshape(C, -1)
        flat /= (np.linalg.norm(flat, axis=0, keepdims=True) + 1e-8)
        return flat.reshape(C, Hf, Wf)
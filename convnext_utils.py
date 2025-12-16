#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convnext_utils.py
- 提供基于 ConvNeXt 的特征提取工具类
- 创新点：多层级特征级联融合 (Pyramid Feature Aggregation)
- 同时利用 Stage 2 (浅层纹理) and Stage 4 (深层语义)
- 支持 extract_batch_features 批量提取
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models


class ConvNeXtFeatureExtractor:
    def __init__(
            self,
            model_name: str = "convnext_tiny",
            img_size: int = 224,
            device: str = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.img_size = int(img_size)

        # ---- 加载模型 ----
        full_model = None
        # 尝试加载预训练权重 (兼容不同 torchvision 版本)
        if model_name == "convnext_tiny":
            try:
                weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
                full_model = models.convnext_tiny(weights=weights)
            except AttributeError:
                try:
                    full_model = models.convnext_tiny(weights="IMAGENET1K_V1")
                except:
                    full_model = models.convnext_tiny(pretrained=True)
        elif model_name == "convnext_small":
            try:
                weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1
                full_model = models.convnext_small(weights=weights)
            except AttributeError:
                try:
                    full_model = models.convnext_small(weights="IMAGENET1K_V1")
                except:
                    full_model = models.convnext_small(pretrained=True)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        # 拆解模型结构：features 包含 [0..7]
        # 0-1: Stem, 2: Stage1, 4: Stage2, 6: Stage3, 7: Stage4
        self.features = full_model.features.to(self.device).eval()

        # 定义提取层级：同时利用浅层和深层
        self.stage_indices = {
            'shallow': 4,  # Stage 2 (下采样 8倍) -> 关注局部纹理
            'deep': 7  # Stage 4 (下采样 32倍) -> 关注全局语义
        }

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
        ])

    def _preprocess_batch(self, bgr_list):
        batch_tensors = []
        for img in bgr_list:
            if img is None or img.size == 0:
                # 异常处理：黑图填充
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

            # 转 RGB 并缩放
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            batch_tensors.append(self.transform(resized))

        if not batch_tensors:
            return None

        return torch.stack(batch_tensors).to(self.device)

    def extract_batch_features(self, bgr_list: list) -> np.ndarray:
        """
        返回融合后的特征向量 [B, C_shallow + C_deep]
        """
        x = self._preprocess_batch(bgr_list)
        if x is None: return np.array([])

        with torch.no_grad():
            # 1. 前向传播到浅层 (Stage 2)
            for i in range(self.stage_indices['shallow'] + 1):
                x = self.features[i](x)

            # 提取浅层特征 & L2归一化
            f_shallow = self.avgpool(x).flatten(1)
            f_shallow = torch.nn.functional.normalize(f_shallow, p=2, dim=1)

            # 2. 继续传播到深层 (Stage 4)
            for i in range(self.stage_indices['shallow'] + 1, self.stage_indices['deep'] + 1):
                x = self.features[i](x)

            # 提取深层特征 & L2归一化
            f_deep = self.avgpool(x).flatten(1)
            f_deep = torch.nn.functional.normalize(f_deep, p=2, dim=1)

            # 3. 特征级联融合 (Concat)
            f_final = torch.cat([f_shallow, f_deep], dim=1)

        f_np = f_final.cpu().numpy().astype("float32")

        # 对整体再做一次归一化，确保余弦相似度计算正确
        norm = np.linalg.norm(f_np, axis=1, keepdims=True) + 1e-8
        f_np /= norm

        return f_np

    def extract_global_feature(self, bgr: np.ndarray) -> np.ndarray:
        """兼容旧接口：单张图片提取"""
        feats = self.extract_batch_features([bgr])
        if len(feats) == 0: return np.array([])
        return feats[0]
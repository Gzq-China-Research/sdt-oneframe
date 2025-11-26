import cv2
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from PIL import Image

class RobustPatchDetector:
    def __init__(self, output_dir="robust_output"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def load_images(self, patch_image_path, no_patch_image_path):
        """加载图像"""
        self.patch_img = cv2.imread(patch_image_path)
        self.no_patch_img = cv2.imread(no_patch_image_path)

        if self.patch_img is None or self.no_patch_img is None:
            raise ValueError("无法加载图像")

        print(f"图像加载成功: {self.patch_img.shape}")
        return True

    def robust_feature_matching(self):
        """鲁棒的特征匹配"""
        # 使用多种特征检测器提高匹配成功率
        detectors = [
            cv2.SIFT_create(1000),  # 增加特征点数量
            cv2.ORB_create(2000)
        ]

        best_H = None
        best_inliers = 0

        for detector in detectors:
            try:
                # 检测关键点和描述符
                kp1, des1 = detector.detectAndCompute(self.patch_img, None)
                kp2, des2 = detector.detectAndCompute(self.no_patch_img, None)

                if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
                    continue

                # 特征匹配
                if isinstance(detector, cv2.SIFT):
                    matcher = cv2.BFMatcher(cv2.NORM_L2)
                else:
                    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

                matches = matcher.knnMatch(des1, des2, k=2)

                # Lowe's比率测试
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)

                if len(good_matches) < 10:
                    continue

                # 提取匹配点
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # 计算单应性矩阵
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if H is not None:
                    inliers = np.sum(mask)
                    if inliers > best_inliers:
                        best_inliers = inliers
                        best_H = H

            except Exception as e:
                print(f"特征匹配失败: {e}")
                continue

        return best_H, best_inliers

    def detect_patches_in_patch_image(self):
        """在有贴片图像中检测贴片"""
        gray = cv2.cvtColor(self.patch_img, cv2.COLOR_BGR2GRAY)

        # 方法1: 使用HoughCircles，但设置更严格的参数
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=50,  # 增加最小距离
            param1=100,  # 增加Canny边缘检测阈值
            param2=40,  # 增加圆心检测阈值
            minRadius=15,
            maxRadius=60
        )

        detected_patches = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

            # 筛选圆形：基于对比度和纹理
            for (x, y, r) in circles:
                # 检查圆形区域与周围区域的对比度
                patch_roi = gray[max(0, y - r):min(gray.shape[0], y + r),
                            max(0, x - r):min(gray.shape[1], x + r)]
                if patch_roi.size == 0:
                    continue

                # 创建圆形掩码
                mask = np.zeros_like(gray)
                cv2.circle(mask, (x, y), r, 255, -1)

                # 计算内部和外部区域的对比度
                inside_mean = cv2.mean(gray, mask=mask)[0]
                outside_mask = cv2.bitwise_not(mask)
                outside_mean = cv2.mean(gray, mask=outside_mask)[0]

                contrast = abs(inside_mean - outside_mean)

                # 如果对比度足够高，认为是贴片
                if contrast > 20:  # 调整这个阈值
                    detected_patches.append((x, y, r))

        print(f"初步检测到 {len(detected_patches)} 个候选贴片")
        return detected_patches

    def refine_patches_by_texture(self, patches):
        """通过纹理分析精化贴片检测"""
        gray = cv2.cvtColor(self.patch_img, cv2.COLOR_BGR2GRAY)
        refined_patches = []

        for (x, y, r) in patches:
            # 提取贴片区域
            patch_roi = gray[max(0, y - r):min(gray.shape[0], y + r),
                        max(0, x - r):min(gray.shape[1], x + r)]

            if patch_roi.size == 0:
                continue

            # 计算贴片区域的纹理特征
            # 1. 计算局部二值模式(LBP)方差
            lbp = cv2.calcHist([patch_roi], [0], None, [256], [0, 256])
            lbp_variance = np.var(lbp)

            # 2. 计算梯度幅度
            grad_x = cv2.Sobel(patch_roi, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(patch_roi, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
            grad_variance = np.var(gradient_magnitude)

            # 3. 计算熵
            hist = cv2.calcHist([patch_roi], [0], None, [256], [0, 256])
            hist = hist / hist.sum()
            entropy = -np.sum(hist * np.log2(hist + 1e-10))

            # 综合纹理特征判断
            texture_score = lbp_variance * 0.4 + grad_variance * 0.4 + entropy * 0.2

            # 如果纹理特征足够显著，保留贴片
            if texture_score > 10:  # 调整这个阈值
                refined_patches.append((x, y, r))

        print(f"纹理分析后保留 {len(refined_patches)} 个贴片")
        return refined_patches

    def map_patches_to_target(self, patches, H):
        """将贴片位置映射到目标图像"""
        if H is None:
            print("没有变换矩阵，无法映射")
            return []

        mapped_patches = []

        for (x, y, r) in patches:
            # 创建源点（贴片中心和边界点）
            src_points = np.array([
                [[x, y]],  # 中心
                [[x - r, y]],  # 左边界
                [[x + r, y]],  # 右边界
                [[x, y - r]],  # 上边界
                [[x, y + r]]  # 下边界
            ], dtype=np.float32)

            # 变换到目标图像
            dst_points = cv2.perspectiveTransform(src_points, H)

            # 计算变换后的中心和半径
            center_x, center_y = dst_points[0][0]

            # 计算半径（取四个方向半径的平均值）
            radii = [
                np.linalg.norm(dst_points[1][0] - dst_points[0][0]),  # 左
                np.linalg.norm(dst_points[2][0] - dst_points[0][0]),  # 右
                np.linalg.norm(dst_points[3][0] - dst_points[0][0]),  # 上
                np.linalg.norm(dst_points[4][0] - dst_points[0][0])  # 下
            ]
            avg_radius = np.mean(radii)

            # 确保位置在图像范围内
            if (0 <= center_x < self.no_patch_img.shape[1] and
                    0 <= center_y < self.no_patch_img.shape[0] and
                    avg_radius > 5):
                mapped_patches.append((int(center_x), int(center_y), int(avg_radius)))

        return mapped_patches

    def visualize_results(self, patches_in_patch, patches_in_no_patch):
        """可视化结果"""
        # 在有贴片图像上显示检测结果
        patch_result = self.patch_img.copy()
        for (x, y, r) in patches_in_patch:
            cv2.circle(patch_result, (x, y), r, (0, 255, 0), 3)
            cv2.circle(patch_result, (x, y), 2, (0, 0, 255), 3)

        # 在无贴片图像上显示映射结果
        no_patch_result = self.no_patch_img.copy()
        for (x, y, r) in patches_in_no_patch:
            cv2.circle(no_patch_result, (x, y), r, (0, 255, 0), 3)
            cv2.circle(no_patch_result, (x, y), 2, (0, 0, 255), 3)

        # 并排显示
        combined = np.hstack([patch_result, no_patch_result])

        # 调整大小以适应屏幕
        h, w = combined.shape[:2]
        scale = min(1800 / w, 800 / h)
        if scale < 1:
            new_w = int(w * scale)
            new_h = int(h * scale)
            combined = cv2.resize(combined, (new_w, new_h))

        cv2.imshow("检测结果 (左:有贴片, 右:无贴片)", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 保存结果
        cv2.imwrite(f"{self.output_dir}/patch_detection.jpg", patch_result)
        cv2.imwrite(f"{self.output_dir}/mapped_result.jpg", no_patch_result)
        cv2.imwrite(f"{self.output_dir}/combined_result.jpg", combined)

    def run_detection(self, patch_image_path, no_patch_image_path):
        """运行检测"""
        try:
            print("=== 鲁棒贴片检测 ===")

            # 1. 加载图像
            self.load_images(patch_image_path, no_patch_image_path)

            # 2. 在有贴片图像中检测贴片
            print("1. 在有贴片图像中检测贴片...")
            patches_in_patch = self.detect_patches_in_patch_image()

            if not patches_in_patch:
                print("无法检测到贴片")
                return []

            # 3. 通过纹理分析精化检测结果
            print("2. 通过纹理分析精化检测结果...")
            refined_patches = self.refine_patches_by_texture(patches_in_patch)

            if not refined_patches:
                print("纹理分析后没有保留任何贴片")
                return []

            # 4. 图像配准
            print("3. 图像配准...")
            H, inliers = self.robust_feature_matching()

            if H is None:
                print("图像配准失败")
                return []

            print(f"配准成功，内点数量: {inliers}")

            # 5. 映射贴片位置
            print("4. 映射贴片位置到无贴片图像...")
            patches_in_no_patch = self.map_patches_to_target(refined_patches, H)

            print(f"映射后得到 {len(patches_in_no_patch)} 个贴片位置")

            # 6. 可视化结果
            print("5. 可视化结果...")
            self.visualize_results(refined_patches, patches_in_no_patch)

            return patches_in_no_patch

        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
            return []


class DeepLearningPatchDetector:
    def __init__(self, output_dir="dl_output"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 加载预训练模型（这里使用一个简单的示例）
        # 实际应用中，您可能需要训练一个专门用于检测贴片的模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def detect_patches_using_circle_detection(self, image):
        """使用圆形检测在有贴片图像中找到贴片"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 使用多种参数进行圆形检测
        all_circles = []

        param2_values = [20, 30, 40]  # 圆心检测阈值
        minDist_values = [30, 50, 70]  # 最小圆心距离

        for param2 in param2_values:
            for minDist in minDist_values:
                circles = cv2.HoughCircles(
                    gray, cv2.HOUGH_GRADIENT, dp=1.2,
                    minDist=minDist, param1=50, param2=param2,
                    minRadius=15, maxRadius=60
                )

                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")
                    all_circles.extend([(x, y, r) for (x, y, r) in circles])

        # 去除重复的检测结果
        unique_circles = self.remove_duplicate_circles(all_circles)

        # 基于对比度筛选
        filtered_circles = []
        for (x, y, r) in unique_circles:
            # 创建圆形掩码
            mask = np.zeros_like(gray)
            cv2.circle(mask, (x, y), r, 255, -1)

            # 计算内部和外部区域的对比度
            inside_mean = cv2.mean(gray, mask=mask)[0]

            # 创建外部区域掩码（环形区域）
            outer_mask = np.zeros_like(gray)
            cv2.circle(outer_mask, (x, y), r + 10, 255, -1)
            outer_mask = cv2.bitwise_xor(outer_mask, mask)

            outside_mean = cv2.mean(gray, mask=outer_mask)[0]

            contrast = abs(inside_mean - outside_mean)

            if contrast > 15:  # 对比度阈值
                filtered_circles.append((x, y, r))

        return filtered_circles

    def remove_duplicate_circles(self, circles, distance_threshold=20):
        """去除重复的圆形检测结果"""
        if not circles:
            return []

        unique_circles = []
        for circle in circles:
            x1, y1, r1 = circle
            is_duplicate = False

            for existing in unique_circles:
                x2, y2, r2 = existing
                distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                if distance < distance_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_circles.append(circle)

        return unique_circles

    def run_detection(self, patch_image_path, no_patch_image_path):
        """运行检测"""
        try:
            print("=== 深度学习辅助贴片检测 ===")

            # 加载图像
            patch_img = cv2.imread(patch_image_path)
            no_patch_img = cv2.imread(no_patch_image_path)

            if patch_img is None or no_patch_img is None:
                print("无法加载图像")
                return []

            # 在有贴片图像中检测贴片
            print("1. 在有贴片图像中检测贴片...")
            patches = self.detect_patches_using_circle_detection(patch_img)

            print(f"检测到 {len(patches)} 个贴片")

            # 可视化结果
            result_img = patch_img.copy()
            for (x, y, r) in patches:
                cv2.circle(result_img, (x, y), r, (0, 255, 0), 3)
                cv2.circle(result_img, (x, y), 2, (0, 0, 255), 3)

            cv2.imshow("贴片检测结果", result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cv2.imwrite(f"{self.output_dir}/detection_result.jpg", result_img)

            return patches

        except Exception as e:
            print(f"错误: {e}")
            return []


class SegmentationBasedDetector:
    def __init__(self, output_dir="seg_output"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def segment_and_detect_patches(self, patch_image):
        """使用图像分割方法检测贴片"""
        # 转换为LAB颜色空间
        lab = cv2.cvtColor(patch_image, cv2.COLOR_BGR2LAB)

        # 使用K-means聚类进行分割
        pixel_values = lab.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)

        # 定义聚类条件
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 4  # 聚类数量

        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # 将像素转换回8位值
        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(patch_image.shape)

        # 对每个聚类进行圆形检测
        all_patches = []

        for i in range(k):
            # 创建当前聚类的掩码
            mask = (labels == i).reshape(patch_image.shape[:2]).astype(np.uint8) * 255

            # 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # 检测轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 100 or area > 5000:
                    continue

                # 计算圆形度
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue

                circularity = 4 * np.pi * area / (perimeter * perimeter)

                if circularity > 0.6:  # 较高的圆形度要求
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    all_patches.append((int(x), int(y), int(radius)))

        # 去除重复的检测结果
        unique_patches = self.remove_duplicate_circles(all_patches)

        return unique_patches

    def remove_duplicate_circles(self, circles, distance_threshold=20):
        """去除重复的圆形检测结果"""
        if not circles:
            return []

        unique_circles = []
        for circle in circles:
            x1, y1, r1 = circle
            is_duplicate = False

            for existing in unique_circles:
                x2, y2, r2 = existing
                distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                if distance < distance_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_circles.append(circle)

        return unique_circles

    def run_detection(self, patch_image_path, no_patch_image_path):
        """运行检测"""
        try:
            print("=== 基于分割的贴片检测 ===")

            # 加载图像
            patch_img = cv2.imread(patch_image_path)

            if patch_img is None:
                print("无法加载图像")
                return []

            # 使用分割方法检测贴片
            print("1. 使用图像分割检测贴片...")
            patches = self.segment_and_detect_patches(patch_img)

            print(f"检测到 {len(patches)} 个贴片")

            # 可视化结果
            result_img = patch_img.copy()
            for (x, y, r) in patches:
                cv2.circle(result_img, (x, y), r, (0, 255, 0), 3)
                cv2.circle(result_img, (x, y), 2, (0, 0, 255), 3)

            cv2.imshow("分割检测结果", result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cv2.imwrite(f"{self.output_dir}/segmentation_result.jpg", result_img)

            return patches

        except Exception as e:
            print(f"错误: {e}")
            return []


def test_all_methods(patch_image_path, no_patch_image_path):
    """测试所有方法"""
    print("=== 测试所有贴片检测方法 ===\n")

    # 方法1: 鲁棒检测器
    print("方法1: 鲁棒特征匹配检测器")
    detector1 = RobustPatchDetector("output_method1")
    result1 = detector1.run_detection(patch_image_path, no_patch_image_path)
    print(f"结果: {len(result1)} 个贴片\n")

    # 方法2: 深度学习辅助检测
    print("方法2: 深度学习辅助检测")
    detector2 = DeepLearningPatchDetector("output_method2")
    result2 = detector2.run_detection(patch_image_path, no_patch_image_path)
    print(f"结果: {len(result2)} 个贴片\n")

    # 方法3: 基于分割的检测
    print("方法3: 基于分割的检测")
    detector3 = SegmentationBasedDetector("output_method3")
    result3 = detector3.run_detection(patch_image_path, no_patch_image_path)
    print(f"结果: {len(result3)} 个贴片\n")

    print("=== 测试完成 ===")
    return result1, result2, result3


# 运行测试
if __name__ == "__main__":
    patch_path = "data/raw/preop_one.png"
    no_patch_path = "data/raw/intraop_one.png"

    results = test_all_methods(patch_path, no_patch_path)
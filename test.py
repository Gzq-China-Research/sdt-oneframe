import cv2
import numpy as np
import argparse


def detect_and_show_circle(image_path):
    # 读取图像（灰度图）
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 预处理（与原代码保持一致，增强检测效果）
    blur = cv2.medianBlur(gray, 5)  # 中值滤波去噪
    # 可选：增强对比度（根据图像情况调整）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    proc = clahe.apply(blur)

    # 霍夫圆检测（参数与原代码一致）
    hint_diam_px = 120  # 直径初值提示（与原代码匹配）
    circles = cv2.HoughCircles(
        blur,  # 输入图像（使用模糊图减少噪声干扰）
        cv2.HOUGH_GRADIENT,  # 检测方法
        dp=1.2,  # 累加器分辨率与图像分辨率的比值
        minDist=int(min(proc.shape) / 5),  # 圆心最小距离
        param1=100,  # Canny边缘检测的高阈值
        param2=20,  # 圆心检测阈值（值越小可能检测到更多圆）
        minRadius=int(0.15 * hint_diam_px),  # 最小半径
        maxRadius=int(1.2 * hint_diam_px)  # 最大半径
    )

    # 绘制检测结果
    if circles is not None:
        # 转换为整数坐标（霍夫返回的是浮点数）
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            cx, cy, r = circle  # 圆心(x,y)和半径r
            # 画圆（红色，线宽2）
            cv2.circle(img, (cx, cy), r, (0, 0, 255), 2)
            # 画圆心（绿色，半径3的实心圆）
            cv2.circle(img, (cx, cy), 3, (0, 255, 0), -1)
            # 显示圆心坐标和半径
            text = f"圆心: ({cx}, {cy}), 半径: {r}px"
            cv2.putText(img, text, (cx - 100, cy - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        print("霍夫圆检测成功，已标记结果")
    else:
        print("霍夫圆检测失败，未找到符合条件的圆")

    # 显示图像
    cv2.namedWindow("霍夫圆检测结果", cv2.WINDOW_NORMAL)
    cv2.imshow("霍夫圆检测结果", img)
    cv2.waitKey(0)  # 按任意键关闭窗口
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 解析命令行参数（指定图像路径）
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="输入带贴片的术前图像路径")
    args = parser.parse_args()
    detect_and_show_circle(args.image)
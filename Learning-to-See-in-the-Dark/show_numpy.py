import os
import numpy as np

# 文件夹路径
folder_path = 'remote_coco/Val/Low_NP'

# 遍历文件夹下的所有文件
for file_name in os.listdir(folder_path):
    # 检查文件扩展名是否为 .npy
    if file_name.endswith('.npy'):
        # 构建文件的完整路径
        file_path = os.path.join(folder_path, file_name)

        # 加载 NumPy 文件
        image_array = np.load(file_path)

        # 打印数组的形状
        print(f"File: {file_name}, Shape: {image_array.shape}")

        # 检查是否是三通道图像（例如，RGB）
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            print(
                f"  This is a three-channel image (e.g., RGB) with dimensions: {image_array.shape[0]}x{image_array.shape[1]}x3")
        else:
            # 可能是灰度图像或其他类型的图像数据
            print(f"  This is not a standard three-channel image or has different dimensions.")
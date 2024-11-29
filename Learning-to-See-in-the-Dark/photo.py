import os
import numpy as np
from PIL import Image  # 使用PIL库读取图片
import glob

# 定义输入和输出目录
low_dir = './remote_coco/Train/Low/'
normal_dir = './remote_coco/Train/Normal/'
low_output_dir = './remote_coco/Train/Low_NP/'
normal_output_dir = './remote_coco/Train/Normal_NP/'

# 创建输出目录（如果不存在）
os.makedirs(low_output_dir, exist_ok=True)
os.makedirs(normal_output_dir, exist_ok=True)

# 获取所有低质量图片文件
low_files = glob.glob(os.path.join(low_dir, '*.png'))
normal_files = glob.glob(os.path.join(normal_dir, '*.png'))

# 遍历低质量图片
for low_file in low_files:
    # 提取文件名中的数字编号
    filename = os.path.basename(low_file)
    number = filename[3:8]  # 取出编号部分，假设格式为 'low00001.png'

    # 构造对应的正常质量图片文件名
    normal_file = os.path.join(normal_dir, f'normal{number}.png')

    # 检查正常质量图片是否存在
    if os.path.exists(normal_file):
        # 读取低质量图片并保存为Numpy格式
        low_image = Image.open(low_file)
        low_array = np.array(low_image)
        low_save_path = os.path.join(low_output_dir, f'{number}.npy')
        np.save(low_save_path, low_array)

        # 读取正常质量图片并保存为Numpy格式
        normal_image = Image.open(normal_file)
        normal_array = np.array(normal_image)
        normal_save_path = os.path.join(normal_output_dir, f'{number}.npy')
        np.save(normal_save_path, normal_array)

        print(f'Saved {low_save_path} and {normal_save_path}')
    else:
        print(f'Matching normal image not found for {low_file}')

# import cv2 as cv
# import numpy as np
# import os
#
# # 输入文件夹和输出文件夹
# input_dir = 'remote_coco/Val/Normal_ARW/'  # 存放 ARW 文件的文件夹
# output_dir = 'remote_coco/Val/Normal_JPG/'  # 输出解码后 JPG 图片的文件夹
# os.makedirs(output_dir, exist_ok=True)
#
# # 图像分辨率 (确保与你的实际 ARW 文件一致)
# height, width = 640, 640
#
# # 读取和处理 ARW 文件
# for file_name in os.listdir(input_dir):
#     if file_name.endswith('.ARW'):
#         input_path = os.path.join(input_dir, file_name)
#         output_path = os.path.join(output_dir, file_name.replace('.ARW', '.jpg'))
#
#         # 从 ARW 文件中读取 Bayer 数据
#         raw = np.fromfile(input_path, dtype=np.uint8)
#         print(f"Processing {file_name}: Raw data size = {raw.size}")
#
#         # 检查数据大小是否符合分辨率要求
#         expected_size = height * width
#         if raw.size < expected_size:
#             print(f"Error: {file_name} data size ({raw.size}) is smaller than expected ({expected_size})!")
#             continue
#         elif raw.size > expected_size:
#             print(f"Warning: {file_name} data size ({raw.size}) is larger than expected ({expected_size}).")
#             raw = raw[:expected_size]  # 剪裁超出的部分
#
#         # 重塑为 Bayer 数据的形状
#         raw = raw.reshape((height, width))
#
#         # 将 Bayer 数据转换为 RGB 图像 (假设 Bayer 模式是 RGGB)
#         img = cv.cvtColor(raw, cv.COLOR_BAYER_RG2BGR)
#
#         # 保存为 JPG 图片
#         cv.imwrite(output_path, img)
#         print(f"Processed {file_name} -> {output_path}")


# import os
# import numpy as np
# import cv2
#
# # 路径设置
# low_dir = './remote_coco/Val/Low'
# normal_dir = './remote_coco/Val/Normal'
# output_low_dir = './remote_coco/Val/Low_ARW'
# output_normal_dir = './remote_coco/Val/Normal_ARW'
#
# # 创建输出目录
# os.makedirs(output_low_dir, exist_ok=True)
# os.makedirs(output_normal_dir, exist_ok=True)
#
# def png_to_arw(input_path, output_path, exposure_time):
#     """
#     将 PNG 图像转换为 .ARW 文件，并按照 RGGB 格式生成 Bayer 格式数据。
#     """
#     img = cv2.imread(input_path, cv2.IMREAD_COLOR)
#     if img is None:
#         raise ValueError(f"无法读取图片文件: {input_path}")
#
#     # 确保图像尺寸为 2 的倍数
#     row, col, _ = img.shape
#     ext_x, ext_y = row % 2, col % 2
#     img = img[0:row-ext_x, 0:col-ext_y, :]
#
#     # 初始化 Bayer 格式图像
#     row, col, _ = img.shape
#     raw = np.zeros((row, col), dtype=np.uint16)
#
#     # 按 RGGB 格式分配像素
#     raw[0:row:2, 0:col:2] = img[0:row:2, 0:col:2, 2]  # R
#     raw[0:row:2, 1:col:2] = img[0:row:2, 1:col:2, 1]  # G1
#     raw[1:row:2, 0:col:2] = img[1:row:2, 0:col:2, 1]  # G2
#     raw[1:row:2, 1:col:2] = img[1:row:2, 1:col:2, 0]  # B
#
#     # 保存为 .ARW 格式
#     with open(output_path, 'wb') as f:
#         # 写入伪头信息
#         f.write(b'FAKE_RAW_HEADER')  # 仅作为示例，实际 .ARW 文件需要更复杂的元数据
#         raw.tofile(f)
#     print(f"保存 .ARW 文件: {output_path}")
#
# # 遍历 Low 和 Normal 文件夹，进行文件转换与命名
# low_files = sorted([f for f in os.listdir(low_dir) if f.endswith('.png')])
# normal_files = sorted([f for f in os.listdir(normal_dir) if f.endswith('.png')])
#
# if len(low_files) != len(normal_files):
#     raise ValueError("Low 和 Normal 文件夹中的文件数量不匹配！")
#
# for i, (low_file, normal_file) in enumerate(zip(low_files, normal_files)):
#     # 提取编号
#     low_id = int(os.path.splitext(low_file)[0][-5:])
#     normal_id = int(os.path.splitext(normal_file)[0][-5:])
#     if low_id != normal_id:
#         raise ValueError(f"编号不匹配: {low_file} 和 {normal_file}")
#
#     # 生成文件名
#     low_output_path = os.path.join(output_low_dir, f"{low_id:05d}_00_0.1s.ARW")
#     normal_output_path = os.path.join(output_normal_dir, f"{low_id:05d}_00_10s.ARW")
#
#     # 转换并保存为 ARW 文件
#     png_to_arw(os.path.join(low_dir, low_file), low_output_path, "0.1s")
#     png_to_arw(os.path.join(normal_dir, normal_file), normal_output_path, "10s")

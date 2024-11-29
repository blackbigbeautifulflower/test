from __future__ import division
import os
import tensorflow as tf
import numpy as np
import glob
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim

# 文件路径
input_dir = './remote_coco/Val/Low_NP/'  # 测试集目录
gt_dir = './remote_coco/Val/Normal_NP/'  # Ground truth目录
checkpoint_dir = './result_Sony/'  # 训练过程中保存的模型路径
result_dir = './result_Sony/'  # 结果保存路径

# 获取测试集ID
test_fns = glob.glob(gt_dir + '/*.npy')
test_ids = [int(os.path.basename(test_fn).split('.')[0]) for test_fn in test_fns]

DEBUG = 0
if DEBUG == 1:
    test_ids = test_ids[:5]

# 激活函数
def lrelu(x):
    return tf.maximum(x * 0.2, x)

# 自定义层：上采样与拼接
class UpsampleAndConcat(tf.keras.layers.Layer):
    def __init__(self, output_channels, **kwargs):
        super(UpsampleAndConcat, self).__init__(**kwargs)
        self.deconv = tf.keras.layers.Conv2DTranspose(output_channels, kernel_size=2, strides=2, padding='same')

    def call(self, x1, x2):
        x1_up = self.deconv(x1)
        return tf.concat([x1_up, x2], axis=-1)

# 网络结构
def network(input_tensor):
    def conv_block(x, filters, scope):
        return tf.keras.layers.Conv2D(filters, kernel_size=3, activation=tf.nn.leaky_relu, padding='same', name=scope)(x)

    # Encoder
    conv1 = conv_block(input_tensor, 32, 'g_conv1')
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)
    conv2 = conv_block(pool1, 64, 'g_conv2')
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)
    conv3 = conv_block(pool2, 128, 'g_conv3')
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(conv3)
    conv4 = conv_block(pool3, 256, 'g_conv4')
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(conv4)
    conv5 = conv_block(pool4, 512, 'g_conv5')

    # Decoder
    up6 = UpsampleAndConcat(256)(conv5, conv4)
    conv6 = conv_block(up6, 256, 'g_conv6')
    up7 = UpsampleAndConcat(128)(conv6, conv3)
    conv7 = conv_block(up7, 128, 'g_conv7')
    up8 = UpsampleAndConcat(64)(conv7, conv2)
    conv8 = conv_block(up8, 64, 'g_conv8')
    up9 = UpsampleAndConcat(32)(conv8, conv1)
    conv9 = conv_block(up9, 32, 'g_conv9')

    # Output layer
    conv10 = tf.keras.layers.Conv2D(3, kernel_size=1, activation=None, name='g_conv10')(conv9)
    output = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, 0, 1))(conv10)
    return output

# 测试代码
with tf.device('/GPU:0'):  # 使用GPU进行推理
    # 创建模型
    input_tensor = tf.keras.Input(shape=(None, None, 3))  # 网络输入为3通道
    output_tensor = network(input_tensor)
    model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)

    # 加载训练好的模型权重
    model.load_weights(os.path.join(checkpoint_dir, 'model_epoch_3000.ckpt'))

    # 创建结果保存目录
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    total_mse = 0
    total_ssim = 0
    count = 0

    for test_id in test_ids:
        in_file = os.path.join(input_dir, f'{test_id:05d}.npy')
        gt_file = os.path.join(gt_dir, f'{test_id:05d}.npy')
        if not os.path.exists(in_file) or not os.path.exists(gt_file):
            continue

        # 读取测试数据
        input_patch = np.load(in_file).astype(np.float32) / 255.0  # 归一化到 [0, 1]
        gt_patch = np.load(gt_file).astype(np.float32)  # Ground truth 可能已归一化

        input_patch = np.expand_dims(input_patch, axis=0)  # 增加batch维度
        # 网络推理
        output = model(input_patch, training=False)
        output = tf.clip_by_value(output[0], 0, 1).numpy()  # 网络输出范围 [0, 1]

        # 保存输出结果（转换为 [0, 255] 并保存为 PNG）
        output_resized = (output * 255).astype(np.uint8)
        Image.fromarray(output_resized).save(os.path.join(result_dir, f'{test_id:05d}_output.png'))

        # Ground truth 转换为 [0, 255]
        gt_patch_resized = gt_patch.astype(np.uint8)

        # 计算评价指标 (MSE 和 SSIM 基于 [0, 255] 范围)
        mse = np.mean((output_resized - gt_patch_resized) ** 2)
        ssim = compare_ssim(output_resized, gt_patch_resized, multichannel=True)

        print(f"Processed {test_id:05d} - MSE: {mse:.4f}, SSIM: {ssim:.4f}")

        total_mse += mse
        total_ssim += ssim
        count += 1

    # 打印平均MSE和SSIM
    if count > 0:
        avg_mse = total_mse / count
        avg_ssim = total_ssim / count
        print(f"Average MSE: {avg_mse:.4f}, Average SSIM: {avg_ssim:.4f}")



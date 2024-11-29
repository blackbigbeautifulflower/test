from __future__ import division
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import glob

# Directories
input_dir = './remote_coco/Train/Low_NP/'  # 输入目录
gt_dir = './remote_coco/Train/Normal_NP/'  # Ground truth目录
checkpoint_dir = './result_Sony/'  # 模型保存目录
result_dir = './result_Sony/'  # 结果保存目录

# Training settings
ps = 512  # Patch size for training
save_freq = 500
learning_rate = 1e-4

# 激活函数
def lrelu(x):
    return tf.maximum(x * 0.2, x)

# 自定义层：上采样与拼接
class UpsampleAndConcat(layers.Layer):
    def __init__(self, output_channels, **kwargs):
        super(UpsampleAndConcat, self).__init__(**kwargs)
        self.output_channels = output_channels
        self.deconv = layers.Conv2DTranspose(output_channels, kernel_size=2, strides=2, padding='same')

    def call(self, x1, x2):
        x1_up = self.deconv(x1)
        return tf.concat([x1_up, x2], axis=-1)

# 网络结构定义
def network(input_tensor):
    def conv_block(x, filters, scope):
        return tf.keras.layers.Conv2D(filters, kernel_size=3, activation=tf.nn.leaky_relu, padding='same', name=scope)(x)

    # Encoder
    conv1 = conv_block(input_tensor, 32, 'g_conv1')
    pool1 = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)

    conv2 = conv_block(pool1, 64, 'g_conv2')
    pool2 = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)

    conv3 = conv_block(pool2, 128, 'g_conv3')
    pool3 = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(conv3)

    conv4 = conv_block(pool3, 256, 'g_conv4')
    pool4 = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(conv4)

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
    conv10 = layers.Conv2D(3, kernel_size=1, activation=None, name='g_conv10')(conv9)  # 修改为3通道
    output = layers.Lambda(lambda x: tf.clip_by_value(x, 0, 1))(conv10)  # 确保输出范围为[0, 1]

    return output

# 数据预处理函数
def preprocess_data(data):
    """Normalize data to range [0, 1]."""
    return data / 255.0 if data.max() > 1 else data

# 单次训练步骤
@tf.function
def train_step(input_patch, gt_patch):
    with tf.GradientTape() as tape:
        output = model(input_patch, training=True)
        # 确保输出大小和 gt 大小一致
        output_resized = tf.image.resize(output, size=gt_patch.shape[1:3])
        loss = loss_fn(gt_patch, output_resized)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# 加载数据集
train_fns = glob.glob(gt_dir + '*.npy')
train_ids = [int(os.path.basename(train_fn).split('.')[0]) for train_fn in train_fns]

# 定义模型
input_tensor = tf.keras.Input(shape=(None, None, 3))  # 输入为3通道RGB
output_tensor = network(input_tensor)
model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)

# 优化器和学习率调整
initial_learning_rate = 1e-4
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
loss_fn = tf.keras.losses.MeanAbsoluteError()

# 训练主循环
for epoch in range(4001):
    epoch_loss = []
    for ind in np.random.permutation(len(train_ids)):
        train_id = train_ids[ind]
        in_file = os.path.join(input_dir, f'{train_id:05d}.npy')
        gt_file = os.path.join(gt_dir, f'{train_id:05d}.npy')

        if not os.path.exists(in_file) or not os.path.exists(gt_file):
            continue

        input_patch = np.load(in_file)
        gt_patch = np.load(gt_file)

        # 数据预处理
        input_patch = preprocess_data(input_patch)
        gt_patch = preprocess_data(gt_patch)

        # 增加 Batch 维度
        input_patch = np.expand_dims(input_patch, axis=0)
        gt_patch = np.expand_dims(gt_patch, axis=0)

        # 确保输入和输出均为3通道
        if input_patch.shape[-1] != 3 or gt_patch.shape[-1] != 3:
            print(f"Skipping ID {train_id} due to invalid channel size.")
            continue

        # 训练
        loss = train_step(input_patch, gt_patch)
        epoch_loss.append(loss.numpy())

    # 打印每轮平均损失
    print(f"Epoch {epoch}, Average Loss: {np.mean(epoch_loss):.4f}")

    # 保存模型
    if epoch % save_freq == 0:
        model.save_weights(os.path.join(checkpoint_dir, f'model_epoch_{epoch}.ckpt'))


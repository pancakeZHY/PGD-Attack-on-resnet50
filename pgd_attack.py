import os
import tensorflow as tf
import numpy as np
from PIL import Image
from model_test import preprocess_image

# 加载 ResNet50 模型
model = tf.keras.applications.ResNet50(weights='imagenet')
# 原始图像文件夹和保存对抗样本的文件夹
image_folder = './img/'
adv_folder = './img_adv/'  

# PGD 攻击
def pgd_attack(model, original_image, original_label, epsilon, alpha, num_iter):
    # 初始化对抗样本
    adv_image = tf.identity(original_image)
    # 开始迭代
    for i in range(1, num_iter + 1):
        # 计算损失和梯度
        with tf.GradientTape() as tape:
            tape.watch(adv_image)
            prediction = model(adv_image)
            loss = tf.keras.losses.sparse_categorical_crossentropy(original_label, prediction)
        gradient = tape.gradient(loss, adv_image)
        gradient = tf.clip_by_value(gradient, -1.0, 1.0) # 梯度裁剪
        # 更新对抗样本
        adv_image = adv_image + alpha * gradient
        adv_image = tf.clip_by_value(adv_image, original_image - epsilon, original_image + epsilon)
        # 确保像素值在 [0, 255] 范围内
        adv_image = tf.clip_by_value(adv_image, 0, 255)  
        # 输出损失和梯度信息，方便调试
        print(f'iter {i}, loss: {loss.numpy()}, max_gradient: {tf.reduce_max(tf.abs(gradient)).numpy()}')
    return adv_image

if __name__ == '__main__':
    for image_name in os.listdir(image_folder):
        # 图像路径和保存路径
        image_path = os.path.join(image_folder, image_name)
        adv_path = os.path.join(adv_folder, image_name)
        # 获取原始图像和原始标签
        original_image = preprocess_image(image_path)
        original_label = tf.argmax(model.predict(original_image)[0]).numpy()
        print(f'original label: {original_label}')
        # 进行攻击
        adversarial_tensor = pgd_attack(model, 
                                        original_image, 
                                        original_label, 
                                        epsilon=0.01, 
                                        alpha=1, 
                                        num_iter=100)
        # 将对抗样本转换为图像并保存
        adv_image = adversarial_tensor.numpy().squeeze()
        adv_image = adv_image.astype(np.uint8)
        adv_image_pil = Image.fromarray(adv_image)
        adv_image_pil.save(adv_path)
        print(f'the Adversarial example of {image_name} has been saved in {adv_path}.')

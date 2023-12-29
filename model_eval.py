import tensorflow as tf
from PIL import Image
import os
from tensorflow.keras.applications.resnet50 import decode_predictions


# 加载模型
model = tf.keras.applications.ResNet50(weights='imagenet')
# 加载ImageNet数据集标签表
with open('labels/ImageNetLabels.txt') as f:
    labels = f.read().splitlines()


# 将图像转换期望输入格式
def preprocess_image(image_path):
    img = Image.open(image_path).resize((224, 224)).convert('RGB')
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array


# 图像分类
def eval(image_dir):
    for filename in os.listdir(image_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_dir, filename)
        # 预处理图像
        img_array = preprocess_image(image_path)
        # 获取模型的预测结果
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions)
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
            print(f"{imagenet_id}: {label} ({score:.2f})")


if __name__ == "__main__":
    print('\noriginal:')
    image_folder = 'img/'
    eval(image_folder)

    print('\nAdversarial:')
    image_adv_folder = 'img_adv/'
    eval(image_adv_folder)

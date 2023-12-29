import tensorflow as tf
from PIL import Image
import os


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
def classify_images(image_dir):
    for filename in os.listdir(image_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_dir, filename)
        # 预处理图像
        img_array = preprocess_image(image_path)
        # 获取模型的预测结果
        predictions = model.predict(img_array)
        label_index = tf.argmax(predictions[0]).numpy()
        predicted_label = labels[label_index]
        print(f'File: {filename}\nPredicted label: {label_index} {predicted_label}')


if __name__ == "__main__":
    print('\noriginal:')
    image_folder = 'img/'
    classify_images(image_folder)

    print('\nAdversarial:')
    image_adv_folder = 'img_adv/'
    classify_images(image_adv_folder)

# 文件说明
'img/'中是原始图像，'img_adv'中是对抗样本；  
'labels/ImageNetLabels.txt'是imagenet数据集的标签，注意索引号应从0开始，即索引号=行数-1；  
'pgd_attack'对原始图像进行PGD攻击，并保存对抗样本；  
'model_test.py'使用预训练的resnet50模型，对所有图像进行分类并输出标签；  
'model_eval.py'输出多个预测的标签及对应概率；  

# 环境配置
python                    3.9  
tensorflow                2.10  
numpy                     1.26  
pillow                    10.0  

# 运行方法
在终端激活对应环境；  
输入'python pgd_attack.py'进行对抗攻击；  
输入'python model_test.py'查看攻击效果，输出结果如下： 
``` 
original:
1/1 [==============================] - 1s 611ms/step
File: airplane.png
Predicted label: 404 airliner
1/1 [==============================] - 0s 72ms/step
File: cat.png
Predicted label: 285 Egyptian cat
1/1 [==============================] - 0s 65ms/step
File: dog.png
Predicted label: 259 Pomeranian

Adversarial:
1/1 [==============================] - 0s 67ms/step
File: airplane.png
Predicted label: 812 space shuttle
1/1 [==============================] - 0s 71ms/step
File: cat.png
Predicted label: 478 carton
1/1 [==============================] - 0s 69ms/step
File: dog.png
Predicted label: 258 Samoyed
```

输入'python model_eval.py'评估攻击效果，输出结果如下： 
``` 
original:
1/1 [==============================] - 1s 636ms/step
n02690373: airliner (0.83)
n04266014: space_shuttle (0.13)
n04552348: warplane (0.02)
n02692877: airship (0.01)
n04008634: projectile (0.00)
1/1 [==============================] - 0s 65ms/step
n02124075: Egyptian_cat (0.27)
n02123045: tabby (0.20)
n02127052: lynx (0.09)
n02971356: carton (0.04)
n02123159: tiger_cat (0.04)
1/1 [==============================] - 0s 74ms/step
n02112018: Pomeranian (0.81)
n02111889: Samoyed (0.15)
n02085620: Chihuahua (0.01)
n02098286: West_Highland_white_terrier (0.01)
n02086079: Pekinese (0.01)

Adversarial:
1/1 [==============================] - 0s 63ms/step
n04266014: space_shuttle (0.99)
n02690373: airliner (0.00)
n04552348: warplane (0.00)
n02692877: airship (0.00)
n03773504: missile (0.00)
1/1 [==============================] - 0s 64ms/step
n02971356: carton (0.45)
n02808440: bathtub (0.13)
n04493381: tub (0.06)
n02123045: tabby (0.03)
n03223299: doormat (0.02)
1/1 [==============================] - 0s 65ms/step
n02111889: Samoyed (0.92)
n02098286: West_Highland_white_terrier (0.06)
n02096177: cairn (0.00)
n02112018: Pomeranian (0.00)
n02114548: white_wolf (0.00)
```
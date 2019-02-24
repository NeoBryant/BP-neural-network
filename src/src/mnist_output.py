# coding: utf-8
import sys
import os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from PIL import Image

# 图片显示
def img_show(img, label):  # 将一个二维数组输出为一张图片
    img = img.reshape(28, 28)
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

    print(label)

# 图片保存
def img_save(isPos, img, label, pre, seq):
    img = img.reshape(28, 28)
    pil_img = Image.fromarray(np.uint8(img))
    if isPos == True:
        path = os.path.abspath(os.path.join(os.getcwd(), ".."))+'/output/positive/'
    else:
        path = os.path.abspath(os.path.join(os.getcwd(), ".."))+'/output/negative/'
    file_name = path+str(seq)+'-label-'+str(label)+"-predict-"+str(pre)+'.png'
    pil_img.save(file_name, 'PNG')

# coding: utf-8

import os
import sys
import numpy as np
from PIL import Image

from BPNetwork import BPNetwork
from load_mnist import load_mnist

sys.path.append(os.pardir)

def read_img(file_name):
    with Image.open(file_name) as image:
        # img -> numpy
        img = np.array(image) # 转化为numpy
        gray_img = np.zeros(shape=(28, 28))
        # rgb转化为单通道
        for i in range(28):
            for j in range(28):
                r, g, b = img[i][j][0], img[i][j][1], img[i][j][2],
                gray_img[i][j] = r * 0.2126 + g * 0.7152 + b * 0.0722

        gray_img = gray_img.flatten()  # 降维
        # 正规化
        gray_img = gray_img.astype(np.float32)
        # gray_img /= 255.0
        gray_img[gray_img != 0] = 1
        #print(gray_img)

    return np.array([gray_img])


def predict(file_name):
    img = read_img(file_name)

    # 权重参数数据文件
    dataset_dir = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/dataset"
    save_file = dataset_dir + "/weights.pkl"

    network = BPNetwork()  # BP神经网络实例
    network.load_weights(save_file)  # 加载训练后的权重参数

    y = network.predict(img)  # 返回识别结果
    num = np.argmax(y)
    return num

if __name__ == "__main__":
    file_name = "../testdata/dst.png"
    # read_img(file_name)
    num = predict(file_name)
    print(num)

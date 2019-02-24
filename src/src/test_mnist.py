# coding: utf-8
import sys
import os
sys.path.append(os.pardir)

import numpy as np
from load_mnist import load_mnist
from BPNetwork import BPNetwork
import pickle
from mnist_output import *

# 获取数据
(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True, flatten=True, one_hot_label=False)

# 权重参数数据文件
dataset_dir = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/dataset"
save_file = dataset_dir + "/weights.pkl"

network = BPNetwork()  # BP神经网络实例
network.load_weights(save_file) # 加载训练后的权重参数

test_acc = network.accuracy(x_test, t_test)
print("\n测试数据集样本数为 10000，识别精确率: ", test_acc, "\n")

test = network.predict(x_test) # 返回识别结果
(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=False, flatten=True, one_hot_label=False)



# 将测试文件数据转化为图片存储
for i in range(len(test)):
    y = np.argmax(test[i])
    if y != t_test[i]: # 识别错误样本
        img_save(False, x_test[i], t_test[i], y, i)
    else: # 识别正确样本
        img_save(True, x_test[i], t_test[i], y, i)


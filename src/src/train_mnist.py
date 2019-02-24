# coding: utf-8
import time
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.pardir)
import numpy as np
from load_mnist import load_mnist
from BPNetwork import BPNetwork
import pickle


# 读入数据
# x_train: [60000][784] 
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = BPNetwork()  # BP神经网络实例

# 超参数
iters_num = 10000 # 训练次数
train_size = x_train.shape[0] # 训练size
batch_size = 100 # 批处理的size，每次训练从训练集中随机选出batch_size数量的样本进行训练
learning_rate = 0.1 # 学习率

train_acc_list = []
test_acc_list = []

print("训练开始..")
start_time = time.time()
for i in range(iters_num): # 训练次数为10000次
    # 获取mini-batch，重复随机梯度下降法
    batch_mask = np.random.choice(train_size, batch_size) # 从train_size个数据中随机选择batch_size个
    x_batch = x_train[batch_mask] # [100][784]
    t_batch = t_train[batch_mask] # [100][10]

    # 计算梯度
    grad = network.gradient(x_batch, t_batch)
    
    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    # 记录每次训练后的精度变化
    '''train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)'''

    # 计算每个epch的识别精度
    if i % 1000 == 0 or i == 9999:
        train_acc = network.accuracy(x_train, t_train)
        print("\t训练", i+1, "次, 训练数据集识别精确率: ", train_acc)
    
end_time = time.time()
print("训练结束! 训练耗时: ", int(end_time-start_time) , "s \n")

# 打印收敛曲线训练集、测试集
'''
index = np.array([i+1 for i in range(iters_num)])
plt.plot(index, train_acc_list)
plt.title("train data set accuracy")
plt.savefig("../output/train_result.png")
plt.show()

plt.plot(index, test_acc_list)
plt.title("test data set accuracy")
plt.savefig("../output/test_result.png")
plt.show()'''

# 将权重参数(W1、b1、W2、b2)存储到pkl中
dataset_dir = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/dataset"
save_file = dataset_dir + "/weights.pkl"

with open(save_file, 'wb') as f:
    pickle.dump(network.params, f, -1)

print("训练后的权重参数数据保存到文件: ", save_file, "\n")

# 测试集精度
'''
test_acc = network.accuracy(x_test, t_test)
print("测试数据集识别精确率: ", test_acc)
'''

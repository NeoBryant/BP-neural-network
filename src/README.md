# BP 神经网络
## 目录
```
.
├── dataset                         //数据集及模型
│   ├── mnist.pkl
│   ├── t10k-images-idx3-ubyte.gz
│   ├── t10k-labels-idx1-ubyte.gz
│   ├── train-images-idx3-ubyte.gz
│   ├── train-labels-idx1-ubyte.gz
│   └── weights.pkl
├── output                          //测试集的识别输出
│   ├── negative
│   └── positive
├── src                             //源文件
│   ├── BPNetwork.py
│   ├── draw_board.py
│   ├── layers.py
│   ├── load_mnist.py
│   ├── mnist_output.py
│   ├── predict_number.py
│   ├── test_mnist.py
│   └── train_mnist.py
└── testdata                         //画版鼠标手写数字输入输出数据
    ├── dst.png
    └── src.png
```

## 环境
操作系统：macOS Mojave；
编程语言：Python3；
编辑器：VS Code；

## 功能
识别手写数字0-9

## 数据集
### mnist数据集
来源：http://yann.lecun.com/exdb/mnist/

### 数据说明
1. 训练集图片 Training set images: train-images-idx3-ubyte.gz (9.9 MB, 解压后 47 MB, 包含 60,000 个样本)
2. 训练集标签 Training set labels: train-labels-idx1-ubyte.gz (29 KB, 解压后 60 KB, 包含 60,000 个标签)
3. 测试集图片 Test set images: t10k-images-idx3-ubyte.gz (1.6 MB, 解压后 7.8 MB, 包含 10,000 个样本)
4. 测试集标签 Test set labels: t10k-labels-idx1-ubyte.gz (5KB, 解压后 10 KB, 包含 10,000 个标签)


## 使用方法
### 训练
src目录下：
```
python train_mnist.py
```
### 测试
```
python test.mnist.py
```
### 鼠标画板手写数字
```
python draw_board.py
```
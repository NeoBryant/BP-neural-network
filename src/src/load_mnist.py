# coding: utf-8
try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path
import gzip
import pickle
import os
import numpy as np

dataset_dir = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/dataset"

save_file = dataset_dir + "/mnist.pkl"

# train_num = 60000
# test_num = 10000
img_size = 784

# 读取gz文件标签数据，并转化为numpy array类型
def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")
    
    return labels

# 读取gz文件图像数据，并转化为numpy array类型
def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")    
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")
    
    return data
    
# 将数据集文件转化为numpy array
def _convert_numpy():
    dataset = {} # 字典str: str
    dataset['train_img'] =  _load_img('train-images-idx3-ubyte.gz')
    dataset['train_label'] = _load_label('train-labels-idx1-ubyte.gz')    
    dataset['test_img'] = _load_img('t10k-images-idx3-ubyte.gz')
    dataset['test_label'] = _load_label('t10k-labels-idx1-ubyte.gz')
    return dataset

def init_mnist(): # 读取gz文件另存为pkl文件
    dataset = _convert_numpy() # 读取数据集gz文件，并转化为numpy array类型
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f: # 将数据集保存为pkl文件
        pickle.dump(dataset, f, -1) 
    print("Done!")

# 将标签int转化为one-hot(eg:[0,0,1,0,0,0,0,0,0,0])数组返回
def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    return T
    

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """读入MNIST数据集
    
    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label : 
        one_hot_label为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    flatten : 是否将图像展开为一维数组
    
    Returns
    -------
    (训练图像, 训练标签), (测试图像, 测试标签)
    """
        
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    
    if normalize: # 将图像的像素值正规化为0.0~1.0
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label: # 标签作为one-hot(eg:[0,0,1,0,0,0,0,0,0,0])数组返回
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])
    
    if not flatten: # 将图像恢复为三维数组，默认输出为一维数组
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label']) 

if __name__ == '__main__':
    init_mnist() # 读取gz文件另存为pkl文件
    

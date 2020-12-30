#!/user/bin/python3
# Author:Confused Pig
# -*- coding: utf-8 -*-

# @Time    : 2020/12/29  16:50
# @Author  : Confused Pig
# @Site    : 
# @File    : main.py
# @Software: PyCharm


import numpy as np
import matplotlib.pyplot as plt
from read_datasets import load_datasets
import h5py

train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes = load_datasets()

'''
#you can see train or test pictures

index = 25
plt.imshow(train_set_x_orig[index])
plt.show()

'''

m_train = train_set_y_orig.shape[1]
m_test = test_set_y_orig.shape[1]
num_px = train_set_x_orig.shape[1]      #train_set_x_orig:(numbers,64,64,3)

#将训练集和测试集所有数据维度都降低并转置。即把他们展开成每张图片的像素占一列
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

#对数据进行标准化，即对每个数据除以255（因为没有像素点的数值会大于255）
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

#数据全部处理完毕，开始构建神经网络。
#       z = wx + b
#       Yi = sigmoid(z)
#       L(Yi,Y) = -Y * log(Yi) - (1-Y) * log(1-Yi)
#       J = (1/m) * sum(L(Yi,Y))           sum表示对L函数累加求和

def sigmoid(x):
    s = 1/(1 + np.exp(-x))
    return s

#initalize w and b
def initalize_parameters_zero(dims):
    '''

    Args:
        dims: 每张图片的像素点数量大小

    Returns:w，b

    '''
    w = np.zeros(shape=(dims,1))
    b = 0

    #确认w维数,b的类型是int or float
    assert (w.shape == (dims,1))
    assert (isinstance(b,float) or isinstance(b,int))

    return w,b


def propagate(w,b,X,Y):
    '''

    Args:
        w: 。。。
        b: 。。。
        X: 输入  类型是(num_px * num_px * 3,1)
        Y: 标签，即 1 或 0

    Returns:
        cost: 逻辑回归的成本
        dw  : w的损失梯度
        db  : b的损失梯度

    '''
    m=X.shape[1]

    #正向传播
    A = sigmoid(np.dot(w.T,X)+b)

    cost = (-1/m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))

    #反向传播
    dw = (1/m) * np.dot(X,(A-Y).T)
    db = (1/m) * np.sum(A-Y)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)                #把shape为1的维度去掉
    assert (cost.shape == ())

    #创建字典保存变量
    grads = {'dw':dw,
             'db':db}

    return cost,grads


def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost=False):

    '''

    Args:
        w: ...
        b: ...
        X: ...
        Y: ...
        num_iterations: 迭代次数
        learning_rate: 更新w，b的学习率
        print_cost: 是否打印

    Returns:
        params: w,b的一个字典
        grads: dw,db的字典
        cost: 用于画学习曲线

    '''
    costs = []
    for i in range(num_iterations):

        cost,grads = propagate(w,b,X,Y)

        dw = grads['dw']
        db = grads['db']

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i%100 ==0:
            costs.append(cost)

        if (print_cost) and (i%100 == 0):
            print('迭代次数:%i ,误差：%f' % (i,cost))

    params = {'w':w,
              'b':b}

    grads = {'dw':dw,
             'db':db}

    return params,grads,costs


def predict(w,b,X):
    '''

    Args:
        w: ...
        b: ...
        X: ...

    Returns:
        Y_predict: 最终预测值，即 0 或 1

    '''
    m = X.shape[1]
    Y_predict = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)

    A = sigmoid(np.dot(w.T,X)+b)

    for i in range(A.shape[1]):
        Y_predict[0,i] = 1 if A[0,i] > 0.5 else 0      #判断阈值是否大于0.5，大于则是1，否则是0

    assert (Y_predict.shape == (1,m))

    return Y_predict


def model(X_train,Y_train,X_test,Y_test,num_iterations=2000,learning_rate=0.5,print_cost=False):

    '''

    Args:
        X_train:
        Y_train:
        X_test:
        Y_test:
        num_iterations:
        learning_rate:
        print_cost:

    Returns:
        dic: 包含有关模型的数据字典

    '''

    w,b = initalize_parameters_zero(X_train.shape[0])
    parameters,grads,costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    w,b = parameters['w'],parameters['b']

    Y_predict_test = predict(w,b,X_test)
    Y_predict_train = predict(w,b,X_train)

    print('训练集准确度： ',format(100 - np.mean(np.abs(Y_predict_train - Y_train)) * 100), '%')
    print('测试集准确度： ', format(100 - np.mean(np.abs(Y_predict_test - Y_test)) * 100), '%')

    dic = {'costs':costs,
           'Y_predict_train':Y_predict_train,
           'Y_predict_test':Y_predict_test,
           'w':w,
           'b':b,
           'learning_rate':learning_rate,
           'num_iterations':num_iterations}

    return dic





#---------------------------------------------Test Model---------------------------------------#
dic = model(train_set_x,train_set_y_orig,test_set_x,test_set_y_orig,num_iterations=2000,learning_rate=0.005,print_cost=True)

#画图
# costs = np.squeeze(dic['costs'])
# plt.plot(costs)
# plt.title('Learning_rate = '+ str(dic['learning_rate']))
# plt.xlabel('iterations (per hundred)')
# plt.ylabel('cost')
# plt.show()

w = dic['w']
b = dic['b']

test = h5py.File('datasets/test_cat.h5','r')
test_cat_x_orig = np.array(test['test_cat'][:])
test_cat_x_flatten = test_cat_x_orig.reshape(test_cat_x_orig.shape[0],-1).T
test_cat = test_cat_x_flatten/255

pre = predict(w,b,test_cat)
print(pre)
from tkinter import  *
from tkinter.simpledialog import *
from tkinter.filedialog import *
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
import csv
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.pipeline import  make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

data = np.loadtxt('D:/yj/mini_project/skin_cancer/skin-cancer-mnist-ham10000.zip/hmnist_28_28_RGB.csv', dtype=np.float32, delimiter=",", skiprows=1, encoding="utf-8")
xdata = data[:, :-1]
ydata = data[:, [-1]]

name = ('D:/yj/mini_project/skin_cancer/skin-cancer-mnist-ham10000.zip/hmnist_28_28_RGB.csv')
name_s=os.path.split(data)

print(name_s[1])

one_hot = sklearn.preprocessing.LabelBinarizer()
ydata = one_hot.fit_transform(ydata)

X_train, X_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.4, random_state=42)

num_filter1 = 32  # 필터의 개수
x = tf.placeholder(tf.float32, shape=[None, 784 * 3])  # 748=28*28*1
x_image = tf.reshape(x, [-1, 28, 28, 3])  # 28*28*1 행렬이 무한개(-1) 있다.

W_conv1 = tf.Variable(tf.random_normal([5, 5, 3, num_filter1]))  # [5,5,1] : 하나의 필터에 대한 행렬&wd의 갯수 num_filter1

h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding="SAME")
b_conv1 = tf.Variable(tf.constant(0.1, shape=[num_filter1]))
h_conv1_cutoff = tf.nn.relu(h_conv1 + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

num_filter2 = 64
W_conv2 = tf.Variable(tf.random_normal([5, 5, num_filter1, num_filter2]))

h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding="SAME")
b_conv2 = tf.Variable(tf.constant(0.1, shape=[num_filter2]))
h_conv2_cutoff = tf.nn.relu(h_conv2 + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

num_units1 = 7 * 7 * num_filter2
num_units2 = 1024

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * num_filter2])

w2 = tf.Variable(tf.random_normal([num_units1, num_units2]))
b2 = tf.Variable(tf.constant(0.2, shape=[num_units2]))

hidden2 = tf.nn.relu(tf.matmul(h_pool2_flat, w2) + b2)

# dropout 함수로 로직이 높은 결과를 만듬
keep_prob = tf.placeholder(tf.float32)
hidden2_drop = tf.nn.dropout(hidden2, keep_prob)

w0 = tf.Variable(tf.zeros([num_units2, 7]))
b0 = tf.Variable(tf.zeros([7]))

k = tf.matmul(hidden2_drop, w0) + b0
p = tf.nn.softmax(k)

t = tf.placeholder(tf.float32, [None, 7])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=k, labels=t))

train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

for i in range(201):
    sess.run(train_step, feed_dict={x: X_train, t: y_train, keep_prob: 0.7})
    # if i % 200 == 0:
    ac = sess.run( accuracy, feed_dict={x: X_test, t: y_test, keep_prob: 1.0})
    print(i, ac)

sess.close()
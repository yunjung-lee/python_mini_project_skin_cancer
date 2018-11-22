

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

def openCSV() :

    global  xdata,ydata,X_train, X_test, y_train, y_test, data

    input_file = askopenfilename(parent=window,filetypes=(("CSV파일", "*.csv"), ("모든파일", "*.*")))
    data = np.loadtxt(input_file, dtype=np.float32, delimiter=",", skiprows=1,encoding="utf-8")


    name = os.path.split(input_file)

    subWindow = Toplevel(window)


    label1 = Label(subWindow, text='CSV파일 이름 -->' + str(name[1]))
    label1.pack()


    subWindow.mainloop()



def one_hot():
    global xdata, ydata,X_train, X_test, y_train, y_test, data
    one_hot = sklearn.preprocessing.LabelBinarizer()
    # if로 분리
    y_train = one_hot.fit_transform(y_train)
    y_test = one_hot.fit_transform(y_test)
    ydata = one_hot.fit_transform(ydata)

    subWindow = Toplevel(window)

    subWindow.geometry('200x100')

    label1 = Label(subWindow, text='One hot encoding 종료')
    label1.pack()

    subWindow.mainloop()

def train_test():
    global xdata, ydata, X_train, X_test, y_train, y_test, data

    xdata = data[:, :-1]
    ydata = data[:, [-1]]

    X_train, X_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.4, random_state=42)

    subWindow = Toplevel(window)


    label1 = Label(subWindow, text='train test split 종료')
    label1.pack()

    subWindow.mainloop()

def deepLearning():
    global xdata, ydata, X_train, X_test, y_train, y_test, data
    x = tf.placeholder(tf.float32, [None, 28 * 28 * 3])
    y = tf.placeholder(tf.float32, [None, 7])
    keep_prob = tf.placeholder(tf.float32)

    w1 = tf.Variable(tf.random_normal([28 * 28 * 3, 256], stddev=0.01))
    L1 = tf.nn.relu(tf.matmul(x, w1))
    L1 = tf.nn.dropout(L1, keep_prob)

    w2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
    L2 = tf.nn.relu(tf.matmul(L1, w2))
    L2 = tf.nn.dropout(L2, keep_prob)

    w3 = tf.Variable(tf.random_normal([256, 7], stddev=0.01))
    model = tf.matmul(L2, w3)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
    prediction = tf.argmax(model, axis=1)
    correct_prediction = tf.equal(prediction, tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)
    for step in range(2001):
        costv, accv, _ = sess.run([cost, accuracy, optimizer],feed_dict={x:X_train,y:y_train,keep_prob:0.8})
        if step % 200 == 0:
            ac = sess.run(accuracy, feed_dict={x: X_test, y: y_test, keep_prob: 1.0})
            print("정확도 : ",ac )
    saver.save(model, 'DL')

    sess.close()

    subWindow = Toplevel(window)

    label1 = Label(subWindow, text='정확도 ->' + str(ac))
    label1.pack()

    subWindow.mainloop()

def cnn():
    global xdata, ydata, X_train, X_test, y_train, y_test, data
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

    for i in range(2001):
        sess.run(train_step, feed_dict={x: X_train, t: y_train, keep_prob: 0.7})
        if i % 200 == 0:
            ac = sess.run(accuracy, feed_dict={x: X_test, t: y_test, keep_prob: 1.0})
            print(i,loss,ac)

    sess.close()

    subWindow = Toplevel(window)

    subWindow.geometry('200x100')

    label1 = Label(subWindow, text='정확도 ->' + str(ac))
    label1.pack()

    subWindow.mainloop()

def knn_cluster():
    global xdata, ydata, X_train, X_test, y_train, y_test, data
    data_df = pd.DataFrame(data)

    ks = range(1,5)
    inertias = []
    for k in ks:
        model = KMeans(n_clusters=k)
        model.fit(data_df)
        inertias.append(model.inertia_)

    plt.plot(ks, inertias, '-o')
    plt.xlabel('number of clusters, k')
    plt.ylabel('intertia')
    plt.xticks(ks)



def saveModel():
    pass

#메인

window = Tk()
window.geometry('800x600')
window.title('Image Ver.0.01 ')



mainMenu = Menu(window)
window.configure(menu = mainMenu)

filemenu = Menu(mainMenu)
mainMenu.add_cascade(label = '파일', menu = filemenu)
filemenu.add_command(label = '열기', command = openCSV)
filemenu.add_command(label = '저장', command = saveModel)

datamenu = Menu(mainMenu)
mainMenu.add_cascade(label = 'data', menu = datamenu)
datamenu.add_command(label = 'One hot encoding', command =one_hot)
datamenu.add_command(label = '트레이닝 데이터 분리', command =train_test )

modelmenu = Menu(mainMenu)
mainMenu.add_cascade(label = 'model', menu = modelmenu)
modelmenu.add_command(label = '신경망 학습', command =deepLearning)
modelmenu.add_command(label = 'CNN 학습', command =cnn)
modelmenu.add_command(label = 'KNN cluster', command =knn_cluster)

window.mainloop()
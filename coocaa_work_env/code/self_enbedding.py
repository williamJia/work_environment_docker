from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# onehot 相关处理
def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


#读入训练数据，形如 123,17,0,1,0,0,1 ##(songid,genreid,user1,user2,user3...) todo
all_df=pd.DataFrame([
    [1,1,1,0],
    [2,2,0,1],
    [3,1,1,0],
    [4,1,1,1],
    [5,1,1,1],
    [6,1,1,0],
    [7,1,1,0],
    [8,1,1,1],
    [9,1,1,0],
    [10,1,1,0],
    [11,1,0,0],
    [12,1,0,0],
    [13,1,1,0]
],columns=['id','genre','user1','user2'])
y_data = to_categorical(np.array(all_df['genre']))
all_df.drop(['genre','id'],axis=1,inplace=True)
x_data = np.array(all_df)


# all_df = pd.read_csv('tags_v3_boolen_data.csv')[:100000]
# all_df.drop(['Unnamed: 0'],axis=1,inplace=True)
# x_data = np.array(all_df)

## 参数
learning_rate = 0.01  ####学习率
training_epochs = 1000  ##训练的周期
batch_size = 256      ##每一批次训练的大小
display_step = 1      ##是否显示计算过程


## 神经网络的参数
n_input = x_data.shape[1]       ## 输入层维度 todo 待完善相关维度
n_hidden_1 = 64               ## 隐层1的神经元个数
n_hidden_2 = 512                ## 隐层2神经元个数
n_hidden_3 = 128                ## 隐层3神经元个数
n_output = 1 # y_data.shape[1]      ## 音乐流派分类数 todo 待完善相关维度

## tf Graph input

X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'softmax_w': tf.Variable(tf.random_normal([n_hidden_2, n_output])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_input])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_1])),
    'softmax_b': tf.Variable(tf.random_normal([n_output])),
}


# def train_self_enbedding(x_data):
#     ## 参数
#     learning_rate = 0.01  ####学习率
#     training_epochs = 1000  ##训练的周期
#     batch_size = 256      ##每一批次训练的大小
#     display_step = 1      ##是否显示计算过程
#
#     ## 神经网络的参数
#     n_input = x_data.shape[1]       ## 输入层维度 todo 待完善相关维度
#     n_hidden_1 = 64               ## 隐层1的神经元个数
#     n_hidden_2 = 512                ## 隐层2神经元个数
#     n_hidden_3 = 128                ## 隐层3神经元个数
#     n_output = 1 # y_data.shape[1]      ## 音乐流派分类数 todo 待完善相关维度
#
#     ## tf Graph input
#     X = tf.placeholder("float", [None, n_input])
#
#     decoder_h1 =  tf.Variable(tf.random_normal([n_hidden_1, n_input]))
#     decoder_b1 = tf.Variable(tf.random_normal([n_input]))
#     encoder_b1 = tf.Variable(tf.random_normal([n_hidden_1]))
#     encoder_h1 = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
#
#     ##************************* 1st hidden layer **************
#     X = tf.placeholder("float", [None, n_input])
#
#     h1_out =tf.nn.sigmoid(tf.add(tf.matmul(X, encoder_h1),
#                                        encoder_b1))
#     keep_prob = tf.placeholder("float")
#     h1_out_drop = tf.nn.dropout(h1_out,keep_prob)
#
#     X_1 = tf.nn.sigmoid(tf.matmul(h1_out_drop,
#                                  decoder_h1)+decoder_b1)
#
#     loss1 = tf.reduce_mean(tf.pow(X - X_1, 2))
#     train_step_1 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss1)
#     sess=tf.Session()
#     sess.run(tf.variables_initializer([encoder_h1,encoder_b1, decoder_h1,decoder_b1]))
#     ## training
#     for i in range(training_epochs):
#         _,c=sess.run([train_step_1,loss1],feed_dict={X:x_data, keep_prob:1.0})
#         if i%5==0:
#             print(c)


decoder_h1 =  tf.Variable(tf.random_normal([n_hidden_1, n_input]))
decoder_b1 = tf.Variable(tf.random_normal([n_input]))
encoder_b1 = tf.Variable(tf.random_normal([n_hidden_1]))
encoder_h1 = tf.Variable(tf.random_normal([n_input, n_hidden_1]))

##************************* 1st hidden layer **************
X = tf.placeholder("float", [None, n_input])

h1_out = tf.nn.sigmoid(tf.add(tf.matmul(X, encoder_h1),
                              encoder_b1))
keep_prob = tf.placeholder("float")
h1_out_drop = tf.nn.dropout(h1_out, keep_prob)

X_1 = tf.nn.sigmoid(tf.matmul(h1_out_drop,
                              decoder_h1) + decoder_b1)

loss1 = tf.reduce_mean(tf.pow(X - X_1, 2))
train_step_1 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss1)
sess = tf.Session()
sess.run(tf.variables_initializer([encoder_h1, encoder_b1, decoder_h1, decoder_b1]))
## training
for i in range(training_epochs):
    _, c = sess.run([train_step_1, loss1], feed_dict={X: x_data, keep_prob: 1.0})
    if i % 5 == 0:
        print(c)

_, c = sess.run([X_1, loss1], feed_dict={X: x_data, keep_prob: 1.0})


x_data_pred = np.array(all_df[:1])
_,c=sess.run([X_1,loss1],feed_dict={X:x_data_pred, keep_prob:1.0})

# # 实际的数值
# for i in x_data_pred[0]:
#     print(i)

# saver = tf.train.Saver()#声明ta.train.Saver()类用于保存
# saver.save(sess,'/home/guanyue/dataming/jiayuepeng/save/a.ckpt')#保存路径为相对路径的save文件夹,保存名为filename.ckpt
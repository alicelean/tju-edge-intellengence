import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
from models.cnn_abstract import ModelCNNAbstract

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class ModelCNNMnist(ModelCNNAbstract):
    def __init__(self):
        super().__init__()
        pass

    def create_graph(self, learning_rate=None):
        '''
        构建graph
        :param learning_rate:
        :return:
        '''
        #图片的特征28*28=784，placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，
        # 它只会分配必要的内存。 等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
        #x代表的是输入图片的浮点数张量，因此定义dtype为"float"。其中，shape的None代表了没有指定张量的shape，
        # 可以feed任何shape的张量，在这里指batch的大小未定。一张mnist图像的大小是2828，784是一张展平的mnist图像的维度，即2828＝784。
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        #标签用10维来表示0-9，10个类别,真值
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])

        #为了使之能用于计算，我们使用reshape将其转换为四维的tensor，其中第一维的－1是指我们可以先不指定，
        # 第二三维是指图像的大小，第四维对应颜色通道数目，灰度图对应1，rgb图对应3.

        self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])

        #第一层卷积,首先在每个5x5网格中，提取出32张特征图。
        # 其中weight_variable中前两维是指网格的大小，第三维的1是指输入通道数目，
        # 第四维的32是指输出通道数目（也可以理解为使用的卷积核个数、得到的特征图张数）。每个输出通道都有一个偏置项，因此偏置项个数为32。
        self.W_conv1 = weight_variable([5, 5, 1, 32])
        self.b_conv1 = bias_variable([32])
        #第二次卷积参数设定
        self.W_conv2 = weight_variable([5, 5, 32, 32])
        self.b_conv2 = bias_variable([32])

        self.W_fc1 = weight_variable([7 * 7 * 32, 256])
        self.b_fc1 = bias_variable([256])
        self.W_fc2 = weight_variable([256, 10])
        self.b_fc2 = bias_variable([10])

        #利用ReLU激活函数，对其进行第一次卷积。
        self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.W_conv1) + self.b_conv1)
        #使用2x2的网格以max pooling的方法池化
        self.h_pool1 = max_pool_2x2(self.h_conv1)
        self.h_norm1 = tf.nn.lrn(self.h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

        #第二次卷积和池化
        self.h_conv2 = tf.nn.relu(conv2d(self.h_norm1, self.W_conv2) + self.b_conv2)
        self.h_norm2 = tf.nn.lrn(self.h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        self.h_pool2 = max_pool_2x2(self.h_norm2)
        #把刚才池化后输出的张量reshape成一个一维向量
        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7 * 7 * 32])

        #再将其与权重相乘，加上偏置项，再通过一个ReLU激活函数
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)
        #应用了简单的softmax，输出。
        self.y = tf.nn.softmax(tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2)

        # 计算交叉熵的代价函数-tf.reduce_sum(self.y_ * tf.log(self.y)，tf.reduce_mean？？
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))

        self.all_weights = [self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2,
                            self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2]

        self._assignment_init()
        #定义优化方法GradientDescentOptimizer和要优化的目标cross_entropy
        self._optimizer_init(learning_rate=learning_rate)
        #计算cross_entropy对指定var_list的梯度
        self.grad = self.optimizer.compute_gradients(self.cross_entropy, var_list=self.all_weights)
        ##找出预测正确的标签
        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        # 得出通过正确个数/总数==准确率
        self.acc = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        #初始化session
        self._session_init()
        self.graph_created = True

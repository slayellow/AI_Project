import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cluster.k_means_ import KMeans

import numpy as np

class ConvModel:
    def __init__(self, sess, name):
        self.labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        self.sess = sess
        self.name = name
        self._build_conv_net()

    def _build_conv_net(self):
        with tf.variable_scope(self.name):
            # Input값을 28x28으로 변환
            self.X = tf.placeholder(tf.float32, [None,784])
            x_img = tf.reshape(self.X, [-1,28,28,1])
            self.Y = tf.placeholder(tf.int64, [None, 10])
            # Conv1 Layer 28x28 --> 14x14
            W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
            L1 = tf.nn.conv2d(x_img, W1, strides=[1, 1, 1, 1], padding='SAME')
            L1 = tf.nn.relu(L1)
            L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # Conv2 Layer 14x14 --> 7x7
            W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
            L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # Fully Connected Layer
            L2 = tf.reshape(L2, [-1, 7 * 7 * 64])
            W3 = tf.get_variable('W3', shape=[7 * 7 * 64, 10], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.random_normal([10]))
            self.hypothesis = tf.matmul(L2, W3) + b
            # Cost, Optimizer Init
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.hypothesis, labels=self.Y))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.cost)
            # Accuracy
            correct_prediction = tf.equal(tf.argmax(self.hypothesis, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test):
        return self.sess.run(self.hypothesis, feed_dict={self.X : x_test})

    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y:y_test})

    def train(self, x_data, y_data):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X:x_data, self.Y: y_data})

class NNModel:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_nn_net()

    def _build_nn_net(self):
        # X : [1,784] / Y : [1,10] Init
        self.X = tf.placeholder(tf.float32, [None, 784])
        self.Y = tf.placeholder(tf.float32, [None, 10])
        # (x*W)+b
        W = tf.Variable(tf.random_normal([784, 10]))
        b = tf.Variable(tf.random_normal([10]))
        self.hypothesis = tf.matmul(self.X,W) + b
        # Cost, Optimizer Init
        self.cost =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.hypothesis, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.cost)
        # Accuracy
        correct_prediction = tf.equal(tf.argmax(self.hypothesis, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test):
        return self.sess.run(self.hypothesis, feed_dict={self.X: x_test})

    def get_accuracy(self, x_test, y_test):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test})

    def train(self, x_data, y_data):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data})

# Train and Test
fm = mnist.read_data_sets('./fashionmnist/',one_hot=True)

batch_size = 100
total_epoch = 10

sess = tf.Session()
nn = NNModel(sess, 'neural_network')
conv = ConvModel(sess, 'conv_network')
models = [nn, conv]

for m_idx, m in enumerate(models):
    sess.run(tf.global_variables_initializer())
    print(m.name, ' : Learning Started!')
    for epoch in range(total_epoch):
        avg_cost = 0
        total_batch = int(fm.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = fm.train.next_batch(batch_size)
            c, _, = m.train(batch_xs, batch_ys)
            avg_cost += c / total_batch
        print('Epoch : ', '%04d'%(epoch+1), 'cost = ','{:.9f}'.format(avg_cost))

    print(m.name, ' Test Accuracy : ', m.get_accuracy(fm.test.images, fm.test.labels))

fig = plt.figure()
labels = nn.predict(fm.test.images)
for i in range(10):
    # 2x5 그리드에 i+1번째 subplot을 추가하고 얻어옴
    subplot = fig.add_subplot(2, 5, i + 1)
    # x, y 축의 지점 표시를 안함
    subplot.set_xticks([])
    subplot.set_yticks([])
    # subplot의 제목을 i번째 결과에 해당하는 숫자로 설정
    subplot.set_title('Pre : ' + str(labels[i].argmax()) + ' Real : ' + str(fm.test.labels[i].argmax()))
    # 입력으로 사용한 i번째 테스트 이미지를 28x28로 재배열하고
    # 이 2차원 배열을 그레이스케일 이미지로 출력
    subplot.imshow(fm.test.images[i].reshape((28, 28)),
                   cmap=plt.cm.gray_r)

plt.show()
plt.close()

fig = plt.figure()
labels = conv.predict(fm.test.images)
for i in range(10):
    # 2x5 그리드에 i+1번째 subplot을 추가하고 얻어옴
    subplot = fig.add_subplot(2, 5, i + 1)
    # x, y 축의 지점 표시를 안함
    subplot.set_xticks([])
    subplot.set_yticks([])
    # subplot의 제목을 i번째 결과에 해당하는 숫자로 설정
    subplot.set_title('Pre : ' + str(labels[i].argmax()) + ' Real : ' + str(fm.test.labels[i].argmax()))
    # 입력으로 사용한 i번째 테스트 이미지를 28x28로 재배열하고
    # 이 2차원 배열을 그레이스케일 이미지로 출력
    subplot.imshow(fm.test.images[i].reshape((28, 28)),
                   cmap=plt.cm.gray_r)

plt.show()

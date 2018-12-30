import tensorflow as tf
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.contrib.factorization import KMeans
from PIL import Image
# from sklearn.cluster import KMeans
from sklearn import mixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


class VAE:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        mu, sigma = self.encoder(self.x)
        self.z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
        self.x_hat = tf.clip_by_value(self.decoder(self.z), 1e-8, 1 - 1e-8)

        # loss
        marginal_likelihood = tf.reduce_sum(self.x * tf.log(self.x_hat) + (1 - self.x) * tf.log(1 - self.x_hat), 1)
        KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)

        marginal_likelihood = tf.reduce_mean(marginal_likelihood)
        KL_divergence = tf.reduce_mean(KL_divergence)

        ELBO = marginal_likelihood - KL_divergence

        self.loss = -ELBO

    def encoder(self, x, n_hidden=1024, dim_z=3):
        with tf.variable_scope("encoder"):
            # initializers
            w_init = tf.contrib.layers.xavier_initializer()
            b_init = tf.constant_initializer(0.)

            # 1st hidden layer
            w0 = tf.get_variable('w0', [x.get_shape()[1], n_hidden], initializer=w_init)
            b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
            h0 = tf.matmul(x, w0) + b0
            h0 = tf.nn.elu(h0)
            h0 = tf.nn.dropout(h0, self.keep_prob)

            # 2nd hidden layer
            w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
            b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
            h1 = tf.matmul(h0, w1) + b1
            h1 = tf.nn.tanh(h1)
            h1 = tf.nn.dropout(h1, self.keep_prob)

            wo = tf.get_variable('wo', [h1.get_shape()[1], dim_z * 2], initializer=w_init)
            bo = tf.get_variable('bo', [dim_z * 2], initializer=b_init)
            gaussian_params = tf.matmul(h1, wo) + bo

            mean = gaussian_params[:, :dim_z]
            stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, dim_z:])

        return mean, stddev

    def decoder(self, z, n_hidden=1024, dim_img=784, reuse=False):
        with tf.variable_scope("decoder", reuse=reuse):
            # initializers
            w_init = tf.contrib.layers.xavier_initializer()
            b_init = tf.constant_initializer(0.)

            # 1st hidden layer
            w0 = tf.get_variable('w0', [z.get_shape()[1], n_hidden], initializer=w_init)
            b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
            h0 = tf.matmul(z, w0) + b0
            h0 = tf.nn.tanh(h0)
            h0 = tf.nn.dropout(h0, self.keep_prob)

            # 2nd hidden layer
            w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
            b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
            h1 = tf.matmul(h0, w1) + b1
            h1 = tf.nn.elu(h1)
            h1 = tf.nn.dropout(h1, self.keep_prob)

            # output layer-mean
            wo = tf.get_variable('wo', [h1.get_shape()[1], dim_img], initializer=w_init)
            bo = tf.get_variable('bo', [dim_img], initializer=b_init)
            y = tf.sigmoid(tf.matmul(h1, wo) + bo)

        return y


def main():
    vae = VAE()
    loss = vae.loss

    # Train parameter
    train_epoch = 100
    train_batch = 64
    learning_rate = 3e-4

    # Train Dataset
    labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    train_dataset = pd.read_csv('./fashionmnist/fashion-mnist_train.csv')
    test_dataset = pd.read_csv('./fashionmnist/fashion-mnist_test.csv')

    train_data = train_dataset[list(train_dataset.columns)[1:]].values / 255
    train_label = train_dataset['label'].values

    test_data = test_dataset[list(test_dataset.columns)[1:]].values / 255
    test_label = test_dataset['label'].values

    dataset = tf.data.Dataset.from_tensor_slices(train_data)
    dataset = dataset.shuffle(100000).repeat().batch(train_batch)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    print(train_data)

    total_batch = int(len(train_data) / train_batch)

    # Train op
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Training Start")
        for epoch in range(train_epoch):
            for i in range(total_batch):
                input_x = sess.run(next_element)
                _, train_loss = sess.run([train_op, loss], feed_dict={vae.x: input_x, vae.keep_prob: 0.7})

            print(train_loss)
            print("학습 진행도 : ", epoch / train_epoch * 100, "%")

        tt = test_data[9000:9001]
        t_out = sess.run(vae.x_hat, feed_dict={vae.x: tt, vae.keep_prob: 1}) * 255
        tt = tt * 255
        print(tt)
        img = Image.fromarray(np.reshape(tt.astype(np.uint8), [28, 28]), "L")
        img.save('ori.jpg')
        print(t_out.astype(np.uint8))
        img = Image.fromarray(np.reshape(t_out.astype(np.uint8), [28, 28]), "L")
        img.save('pre.jpg')

        # make test to latent vector
        test_data = sess.run(vae.z, feed_dict={vae.x: test_data, vae.keep_prob: 1})

    # Kmeans
    k_means_X = tf.placeholder(tf.float32, shape=[None, 3])
    k_means_Y = tf.placeholder(tf.int32, shape=[None, ])
    one_hot_Y = tf.squeeze(tf.one_hot(k_means_Y, 10))
    gmm = KMeans(inputs=k_means_X, num_clusters=64, distance_metric='cosine',
                 use_mini_batch=True)

    # Build KMeans graph
    training_graph = gmm.training_graph()

    if len(training_graph) > 6:  # Tensorflow 1.4+
        (all_scores, cluster_idx, scores, cluster_centers_initialized,
         cluster_centers_var, init_op, train_op) = training_graph
    else:
        (all_scores, cluster_idx, scores, cluster_centers_initialized,
         init_op, train_op) = training_graph

    cluster_idx = cluster_idx[0]  # fix for cluster_idx being a tuple
    avg_distance = tf.reduce_mean(scores)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer(), feed_dict={k_means_X: test_data})
        sess.run(init_op, feed_dict={k_means_X: test_data})

        # Training
        for i in range(1, 1001):
            _, d, idx = sess.run([train_op, avg_distance, cluster_idx],
                                 feed_dict={k_means_X: test_data})
            if i % 10 == 0 or i == 1:
                print("Step %i, Avg Distance: %f" % (i, d))

        counts = np.zeros(shape=(64, 10))
        for i in range(len(idx)):
            counts[idx[i]][test_label[i]] += 1
        # Assign the most frequent label to the centroid
        labels_map = [np.argmax(c) for c in counts]
        print(labels_map)
        labels_map = tf.convert_to_tensor(labels_map)

        # Evaluation ops
        # Lookup: centroid_id -> label
        cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)

        correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(one_hot_Y, 1), tf.int32))
        accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        y_predict = sess.run(cluster_label, feed_dict={k_means_X: test_data})

        print("Test Accuracy:", sess.run(accuracy_op, feed_dict={k_means_X: test_data, k_means_Y: test_label}))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = ['black', 'red', 'blue', 'green', 'yellow', 'purple', 'gray', 'brown', 'cyan', 'magenta']
        for i in range(10):
            px = test_data[:, 0][test_label == i]
            py = test_data[:, 1][test_label == i]
            pz = test_data[:, 2][test_label == i]
            ax.scatter(px, py, pz, c=colors[i], label=labels[i])
        #    ax.scatter(px,py, c=colors[i],label=labels[i])
        plt.legend()
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        plt.savefig('VOLTAGE_ABC_No_MinmaxScalar.png')
        plt.show()
        plt.close()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = ['black', 'red', 'blue', 'green', 'yellow', 'purple', 'gray', 'brown', 'cyan', 'magenta']
        for i in range(10):
            px = test_data[:, 0][y_predict == i]
            py = test_data[:, 1][y_predict == i]
            pz = test_data[:, 2][y_predict == i]
            ax.scatter(px, py, pz, c=colors[i], label=labels[i])
        #    ax.scatter(px,py, c=colors[i],label=labels[i])
        plt.legend()
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        plt.savefig('VOLTAGE_ABC_No_MinmaxScalar.png')
        plt.show()
        plt.close()

    # gmm = mixture.GaussianMixture(n_components=10, covariance_type='diag')
    # gmm.fit(test_data)
    # y_predict = gmm.predict(test_data)
    #
    # counts = np.zeros(shape=(10, 10))
    # for i in range(len(y_predict)):
    #     counts[y_predict[i]][test_label[i]] += 1
    # # Assign the most frequent label to the centroid
    # labels_map = [np.argmax(c) for c in counts]
    # # print(labels_map)
    #
    # for i in range(len(y_predict)):
    #     y_predict[i] = labels_map[y_predict[i]]
    #
    # print(y_predict[0:20])
    # print(list(test_label[0:20]))
    #
    # print(accuracy_score(test_label, y_predict))
    #
    # # estimator = PCA(n_components=3)
    # # X_pca = estimator.fit_transform(train_data)
    # #
    # # gmm = KMeans(n_clusters=10, random_state=1, max_iter=5)
    # # gmm.fit(X_pca)
    # # # y_predict = gmm.predict(X_pca)
    # #
    # # test_data = test_dataset.iloc[:, 1:]
    # # test_label = test_dataset.iloc[:, 0]
    # #
    # # est = PCA(n_components=3)
    # # X_test_pca = est.fit_transform(test_data)
    # # y_predict = gmm.predict(X_test_pca)
    # #
    # # counts = np.zeros(shape=(10, 10))
    # # for i in range(len(y_predict)):
    # #     counts[y_predict[i]][test_label[i]] += 1
    # # # Assign the most frequent label to the centroid
    # # labels_map = [np.argmax(c) for c in counts]
    # # # print(labels_map)
    # #
    # # for i in range(len(y_predict)):
    # #     y_predict[i] = labels_map[y_predict[i]]
    # #
    # # print(y_predict[0:20])
    # # print(list(test_label[0:20]))
    # #
    # # print(accuracy_score(test_label, y_predict))
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # colors = ['black', 'red', 'blue', 'green', 'yellow', 'purple', 'gray', 'brown', 'cyan', 'magenta']
    # for i in range(10):
    #     px = test_data[:, 0][test_label == i]
    #     py = test_data[:, 1][test_label == i]
    #     pz = test_data[:, 2][test_label == i]
    #     ax.scatter(px, py, pz, c=colors[i], label=labels[i])
    # #    ax.scatter(px,py, c=colors[i],label=labels[i])
    # plt.legend()
    # ax.set_xlabel('First Principal Component')
    # ax.set_ylabel('Second Principal Component')
    # plt.savefig('VOLTAGE_ABC_No_MinmaxScalar.png')
    # plt.show()
    # plt.close()
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # colors = ['black', 'red', 'blue', 'green', 'yellow', 'purple', 'gray', 'brown', 'cyan', 'magenta']
    # for i in range(10):
    #     px = test_data[:, 0][y_predict == i]
    #     py = test_data[:, 1][y_predict == i]
    #     pz = test_data[:, 2][y_predict == i]
    #     ax.scatter(px, py, pz, c=colors[i], label=labels[i])
    # #    ax.scatter(px,py, c=colors[i],label=labels[i])
    # plt.legend()
    # ax.set_xlabel('First Principal Component')
    # ax.set_ylabel('Second Principal Component')
    # plt.savefig('VOLTAGE_ABC_No_MinmaxScalar.png')
    # plt.show()
    # plt.close()

    print("END")


if __name__ == '__main__':
    main()

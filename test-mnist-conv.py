"""
Test a deep neural network with 2 rectified convolutional layers followed by 2x2 max pool layers
connect to a fully densely connected 150-hidden rectified units layer then dropout some of them.
Next is readout layer that produces logits based on the remaining alive hidden units
we then apply softmax layer to make them into multinomial probabilities

The training accuracy of this experiment is about 0.97 but we can improve it further by training 20 times more epoch
and increase the number of hidden units to be around 1000 units, lower the learning rate to 1e-4
the accuracy is then 0.992
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, [1, 1, 1, 1], 'SAME')


def maxpool2x2(x):
    return tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print('train:', mnist.train.images.shape, 'test:', mnist.test.images.shape, 'validation:',
      mnist.validation.images.shape)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
print("h_conv1 =", h_conv1)
h_pool1 = maxpool2x2(h_conv1)
print("h_pool1 =", h_pool1)

w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
print("h_conv2 =", h_conv2)
h_pool2 = maxpool2x2(h_conv2)
print("h_pool2 =", h_pool2)

w_fc1 = weight_variable([7 * 7 * 64, 150])
b_fc1 = bias_variable([150])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
print("h_pool2_flat =", h_pool2_flat)
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
print("h_fc1 =", h_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)
print("h_fc1_dropout =", h_fc1_dropout)

w_fc2 = weight_variable([150, 10])
b_fc2 = bias_variable([10])
logits = tf.matmul(h_fc1_dropout, w_fc2) + b_fc2
print('Logits:', logits)
y = tf.nn.softmax(logits)

cross_entropy = -tf.reduce_sum(tf.log(y) * y_, [1])
loss = tf.reduce_mean(cross_entropy)

# region Alternative Quadratic cost function
# squared_error = tf.reduce_sum(tf.squared_difference(y, y_), [1])
# loss = tf.reduce_mean(squared_error)
# endregion

optimizer = tf.train.AdamOptimizer(5e-3)
train_step = optimizer.minimize(loss)

correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print('Training...')
    for i in range(1000):
        batch = mnist.train.next_batch(55)
        sess.run(train_step, {x: batch[0], y_: batch[1], keep_prob: 0.7})
        if i % 25 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
    print('Done.')
    print('Testing...')
    acc = sess.run(accuracy, {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1})
    print('Accuracy on Test Set:', acc)

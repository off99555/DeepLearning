"""
Test a simple linear multinomial logistic regression with softmax activation
accuracy is around 0.92
if we use sigmoid instead, the accuracy is only 0.6767
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print('train:', mnist.train.images.shape, 'test:', mnist.test.images.shape, 'validation:',
      mnist.validation.images.shape)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

logits = tf.matmul(x, w) + b
print('Logits:', logits)
y = tf.nn.softmax(logits)
# y = tf.nn.sigmoid(logits)

cross_entropy = -tf.reduce_sum(tf.log(y) * y_, [1])
loss = tf.reduce_mean(cross_entropy)

# region Alternative Quadratic cost function
# squared_error = tf.reduce_sum(tf.squared_difference(y, y_), [1])
# loss = tf.reduce_mean(squared_error)
# endregion

optimizer = tf.train.GradientDescentOptimizer(0.3)
train_step = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print('Training...', end=' ')
    for i in range(2000):
        batch = mnist.train.next_batch(55)
        sess.run(train_step, {x: batch[0], y_: batch[1]})
    print('Done.')
    weight, bias, train_loss = sess.run([w, b, loss], {x: mnist.train.images, y_: mnist.train.labels})
    print("train_loss =", train_loss)
    print("weight.mean() =", weight.mean(), 'std =', weight.std())
    print("bias.mean() =", bias.mean(), 'std =', bias.std())
    correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    acc = sess.run(accuracy, {x: mnist.test.images, y_: mnist.test.labels})
    print('Accuracy on Test Set:', acc)

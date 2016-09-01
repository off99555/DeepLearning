from tensorflow.contrib import learn
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets('MNIST_data')
print('train:', mnist.train.images.shape, 'test:', mnist.test.images.shape, 'validation:',
      mnist.validation.images.shape)

x_train, y_train = mnist.train.images, mnist.train.labels.astype(np.int32)
x_test, y_test = mnist.test.images, mnist.test.labels.astype(np.int32)

model = learn.DNNClassifier([1000, 500, 200], n_classes=10)
model.fit(x_train, y_train, steps=1000, batch_size=55)

evaluation = model.evaluate(x_test, y_test)
print('evaluation =', evaluation)

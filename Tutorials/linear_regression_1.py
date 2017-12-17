'''
A linear regression learning algorithm example using TensorFlow library.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

# https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/get_started/regression/linear_regression.py

from __future__ import print_function

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 100

# Training Data
n_samples = 50
start_x = 0
end_x = 3.1415 * 2
mean = 0
sigma = 0.5

GT_A = 10
GT_f = 1
GT_phase=0
GT_shift=8

noise = numpy.random.normal(mean, sigma, n_samples)

train_X = numpy.linspace(start_x, end_x, n_samples)
train_Y = GT_A * numpy.sin(GT_f * (train_X + noise) + GT_phase) + GT_shift
train_Y = train_Y

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
A = tf.Variable(rng.randn(), name="amplitude")
f = tf.Variable(rng.randn(), name="frequency")
phase = tf.Variable(rng.randn(), name="phase")
shift = tf.Variable(rng.randn(), name="shift")

# Construct a linear model

# Linear Mode
# pred = tf.add(tf.multiply(X, W), b)

# Sin model
# y = A * sin(f * x + phase) + shift
pred = tf.add( tf.multiply( tf.sin( tf.add( tf.multiply( X, f ), phase ) ), A ), shift )

# inner = tf.add(tf.multiply(f,X), phase)
# pred = tf.add(tf.multiply(inner, A), shift)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

def print_sess(sess, train_X, train_Y, A, f, phase, shift):
    c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
    print("Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(c), \
        "A =", sess.run(A),\
        "f =", sess.run(f),\
        "phase =", sess.run(phase),\
        "shift =", sess.run(shift),\
        )

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print_sess(sess, train_X, train_Y, A, f, phase, shift)

    print("Optimization Finished!")
    print_sess(sess, train_X, train_Y, A, f, phase, shift)
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})

    # Graphic display
    plt.figure("Training Data")
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(pred, feed_dict={X: train_X}), label='Fitted line')
    plt.legend()
    # plt.show()

    # Testing example, as requested (Issue #2)
    # test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    # test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    test_n_samples = 10

    test_X = numpy.random.uniform(start_x, end_x, test_n_samples)
    test_Y = GT_A * numpy.sin(GT_f * (test_X) + GT_phase) + GT_shift


    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))
    
    plt.figure("Test Data")
    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(pred, feed_dict={X: train_X}), label='Fitted line')
    plt.legend()
    plt.show()
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import cross_validation
from sklearn import datasets
from sklearn import metrics

import argparse
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import learn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NUM_PLOT_BINS = 30
MODEL_NAME = 'iris_model'
WEIGHTS_NAME = MODEL_NAME + '/Stack/fully_connected_1/weights'


def model(features, target):
    global args

    regularizer = None
    regularization_type = args.regularization_type.lower()
    regularization_value = args.regularization_value
    if regularization_type == "l1":
        print("Using L1 regularizer, val =", regularization_value)
        regularizer = tf.contrib.layers.l1_regularizer(regularization_value)
    elif regularization_type == "l2":
        print("Using L2 regularizer, val =", regularization_value)
        regularizer = tf.contrib.layers.l2_regularizer(regularization_value)
    else:
        print("Not using regularization")

    target = tf.one_hot(target, 3, 1, 0)
    with tf.variable_scope(MODEL_NAME, regularizer=regularizer):
        features = layers.stack(features, layers.fully_connected, [10, 20, 10])
        logits = layers.fully_connected(features, 3, activation_fn=None)
    loss = tf.contrib.losses.softmax_cross_entropy(logits, target)
    if regularizer:
        loss = loss + sum(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    train_op = tf.contrib.layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer='Adagrad',
        learning_rate=0.1)

    return ({
        'class': tf.argmax(logits, 1),
        'prob': tf.nn.softmax(logits)
    }, loss, train_op)


def plot_weights(flat_weights, plot_file_name, title_name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.suptitle("Weights histogram (1st layer fc) " + title_name)
    ax.hist(flat_weights, NUM_PLOT_BINS, color='green', alpha=0.8)
    print("Saving histogram of weights in:", plot_file_name)
    fig.savefig(plot_file_name)
    plt.close(fig)


def main(argv):
    global args

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--regularization_type',
        default="none",
        help="Regularization type: l1, l2")
    parser.add_argument(
        '--regularization_value',
        type=float,
        default=0.0,
        help="Value used for regularization. defualt 0.0")
    parser.add_argument(
        '--weights_file',
        default='weights_hist.png',
        help="Filename to save the histogram. Default: weights_hist.png")
    args = parser.parse_args()
    iris = datasets.load_iris()
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(
        iris.data, iris.target, test_size=0.2)
    classifier = learn.Estimator(model_fn=model)
    classifier.fit(x_train, y_train, steps=1000)
    y_predicted = [
        p['class'] for p in classifier.predict(
            x_test, as_iterable=True)
    ]
    score = metrics.accuracy_score(y_test, y_predicted)
    print('Accuracy: {0:f}'.format(score))

    weights = classifier.get_variable_value(WEIGHTS_NAME)
    flat_weights = [w for wl in weights for w in wl]
    plot_weights(flat_weights, args.weights_file, args.regularization_type)


if __name__ == '__main__':
    tf.app.run()

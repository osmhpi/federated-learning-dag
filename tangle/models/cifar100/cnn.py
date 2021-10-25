import tensorflow as tf

from ..model import Model
from ..utils.tf_utils import graph_size
from ..baseline_constants import ACCURACY_KEY
from ...lab.dataset import batch_data
import numpy as np


IMAGE_SIZE = 32
DROPOUT = 0.3


class ClientModel(Model):
    def __init__(self, seed, lr, num_classes, optimizer=None):
        self.num_classes = num_classes
        super(ClientModel, self).__init__(seed, lr)

    def create_model(self):
        """Model function for CNN."""
        features = tf.placeholder(
            tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name='features')
        labels = tf.placeholder(tf.int64, shape=[None], name='labels')

        conv1 = tf.layers.conv2d(
            inputs=features,
            filters=32,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=128,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

        flatten = tf.layers.flatten(pool3)
        dense1 = tf.layers.dense(inputs=flatten, units=256, activation=tf.nn.relu)
        dense2 = tf.layers.dense(inputs=dense1, units=128, activation=tf.nn.relu)

        logits = tf.layers.dense(inputs=dense2, units=self.num_classes)
        predictions = {
          "classes": tf.argmax(input=logits, axis=1),
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        train_op = self.optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
        conf_matrix = tf.math.confusion_matrix(labels, predictions["classes"], num_classes=self.num_classes)
        return features, labels, train_op, eval_metric_ops, conf_matrix, loss

    def process_x(self, raw_x_batch):
        return np.array(raw_x_batch)

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)

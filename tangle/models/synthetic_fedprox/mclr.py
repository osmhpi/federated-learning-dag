"""Bag-of-words logistic regression."""

import numpy as np
import os
import sys
import tensorflow as tf

from ..model import Model
# from utils.model_utils import batch_data


class ClientModel(Model):

    def __init__(self, seed, lr, num_classes):
        self.num_classes = num_classes
        super(ClientModel, self).__init__(seed, lr)

    def create_model(self):
        features = tf.placeholder(tf.float32, shape=[None, 60], name='features')
        labels = tf.placeholder(tf.int64, shape=[None,], name='labels')
        logits = tf.layers.dense(inputs=features, units=self.num_classes, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.001))
        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        grads_and_vars = self.optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())
        eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
        conf_matrix = tf.math.confusion_matrix(labels, predictions["classes"], num_classes=self.num_classes)

        return features, labels, train_op, eval_metric_ops, conf_matrix, loss

    def process_x(self, raw_x_batch):
        return np.array(raw_x_batch)

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)

    # def _run_epoch(self, data, batch_size, num_batches):
    #     for batched_x, batched_y in batch_data(data, batch_size, num_batches, self.seed):
    #         input_data = self.process_x(batched_x)
    #         target_data = self.process_y(batched_y)

    #         with self.graph.as_default():
    #             self.sess.run(
    #                 self.train_op,
    #                 feed_dict={
    #                     self.features: input_data,
    #                     self.labels: target_data})

    # def _test(self, data):
    #     x_vecs = self.process_x(data['x'])
    #     labels = self.process_y(data['y'])

    #     with self.graph.as_default():
    #         tot_acc, loss = self.sess.run(
    #             [self.eval_metric_ops, self.loss],
    #             feed_dict={
    #                 self.features: x_vecs,
    #                 self.labels: labels
    #             })

    #     acc = float(tot_acc) / len(x_vecs)
    #     return {'accuracy': acc}

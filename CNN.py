import sys, os, getopt
import tensorflow as tf
from utils import *
from logmanager import *
import math
import numpy as np
import matplotlib.pyplot as plt

batch_size = 10
num_steps = 5000
learning_rate = 0.01
data_showing_step = 500
log_location = '/tmp/alex_nn_log'

SEED = 90320

patch_size = 10
depth_inc = 4
num_hidden_inc = 963
dropout_prob = 0.8
conv_layers = 3
stddev = 0.01
binary = False

dataset, image_size, num_of_classes, num_channels = prepare_data()

dataset = reformat(dataset, image_size, num_channels, num_of_classes, flatten=False)

def accuracy(predictions, labels):
    # print("prediciton ", predictions, "\nlabels ",labels)
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


def nn_model(data, weights, biases, TRAIN=False) :
    with tf.name_scope('Layer_1') as scope:
        conv = tf.nn.conv2d(data, weights['conv1'], strides=[1, 1, 1, 1], padding='SAME', name='conv1')
        bias_add = tf.nn.bias_add(conv, biases['conv1'], name='bias_add_1')
        relu = tf.nn.relu(bias_add, name='relu_1')
        max_pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=scope)

    # print("shape ", max_pool.get_shape())
    with tf.name_scope('Layer_2') as scope:
        conv = tf.nn.conv2d(max_pool, weights['conv2'], strides=[1, 1, 1, 1], padding='SAME', name='conv2')
        bias_add = tf.nn.bias_add(conv, biases['conv2'], name='bias_add_2')
        relu = tf.nn.relu(bias_add, name='relu_2')
        max_pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=scope)

    # print("shape ", max_pool.get_shape())
    with tf.name_scope('Layer_3') as scope:
        conv = tf.nn.conv2d(max_pool, weights['conv3'], strides=[1, 1, 1, 1], padding='SAME', name='conv3')
        bias_add = tf.nn.bias_add(conv, biases['conv3'], name='bias_add_3')
        relu = tf.nn.relu(bias_add, name='relu_3')
        max_pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=scope)
        if TRAIN:
            max_pool = tf.nn.dropout(max_pool, dropout_prob, seed=SEED, name='dropout')

    shape = max_pool.get_shape().as_list()
    # print("shape ",max_pool.get_shape())
    reshape = tf.reshape(max_pool, [shape[0],shape[1] * shape[2] * shape[3]])
    # print("reshape ",reshape.get_shape())
    with tf.name_scope('FC_Layer_1') as scope:
        matmul = tf.matmul(reshape, weights['fc1'], name='fc1_matmul')
        bias_add = tf.nn.bias_add(matmul, biases['fc1'], name='fc1_bias_add')
        relu = tf.nn.relu(bias_add, name=scope)

    with tf.name_scope('FC_Layer_2') as scope:
        matmul = tf.matmul(relu, weights['fc2'], name='fc2_matmul')
        layer_fc2 = tf.nn.bias_add(matmul, biases['fc2'], name=scope)

    return layer_fc2


def fc_first_layer_dimen(image_size, layers) :
    output = image_size
    for x in range(layers):
        output = math.ceil(output/2.0)
    return int(output)

graph = tf.Graph()
with graph.as_default():
    weights = {
        'conv1' : tf.Variable(tf.truncated_normal(shape=[patch_size, patch_size, num_channels, depth_inc],
                    dtype=tf.float32, stddev=stddev, seed=SEED), name='weights_conv1'),
        'conv2' : tf.Variable(tf.truncated_normal(shape=[patch_size, patch_size, depth_inc, depth_inc],
                    dtype=tf.float32, stddev=stddev, seed=SEED), name='weights_conv2'),
        'conv3' : tf.Variable(tf.truncated_normal(shape=[patch_size, patch_size, depth_inc, depth_inc],
                    dtype=tf.float32, stddev=stddev, seed=SEED), name='weights_conv3'),
        'fc1' : tf.Variable(tf.truncated_normal([(fc_first_layer_dimen(image_size, conv_layers) ** 2) * depth_inc,
                    num_hidden_inc], dtype=tf.float32, stddev=stddev), name='weights_fc1'),
        'fc2' : tf.Variable(tf.truncated_normal([num_hidden_inc, num_of_classes], dtype=tf.float32,
                    stddev=stddev, seed=SEED), name='weights_fc2')
    }

    biases = {
        'conv1': tf.Variable(tf.zeros(shape=[depth_inc], dtype=tf.float32), name='biases_conv1'),
        'conv2': tf.Variable(tf.zeros(shape=[depth_inc], dtype=tf.float32), name='biases_conv2'),
        'conv3': tf.Variable(tf.zeros(shape=[depth_inc], dtype=tf.float32), name='biases_conv3'),
        'fc1': tf.Variable(tf.zeros(shape=[num_hidden_inc], dtype=tf.float32), name='biases_fc1'),
        'fc2': tf.Variable(tf.zeros(shape=[num_of_classes], dtype=tf.float32), name='biases_fc2')
    }

    tf.summary.histogram('conv1_weights', weights['conv1'])
    tf.summary.histogram('conv1_biases', biases['conv1'])
    tf.summary.histogram('conv2_weights', weights['conv2'])
    tf.summary.histogram('conv2_biases', biases['conv2'])
    tf.summary.histogram('conv3_weights', weights['conv3'])
    tf.summary.histogram('conv3_biases', biases['conv3'])
    tf.summary.histogram('fc1_weights', weights['fc1'])
    tf.summary.histogram('fc1_biases', biases['fc1'])
    tf.summary.histogram('fc2_weights', weights['fc2'])
    tf.summary.histogram('fc2_biases', biases['fc2'])


    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels), name='TRAIN_DATASET')
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_of_classes), name='TRAIN_LABEL')

    if binary :
        # tf_valid_dataset = tf.constant(dataset.bin_valid_dataset, shape=(10, image_size, image_size, num_channels), name='VALID_DATASET')
        tf_test_dataset = tf.constant(dataset.bin_test_dataset, shape=(10*num_of_classes, image_size, image_size, num_channels), name='TEST_DATASET')
    else :
        # tf_valid_dataset = tf.constant(dataset.grey_valid_dataset, shape=(10, image_size, image_size, num_channels), name='VALID_DATASET')
        tf_test_dataset = tf.constant(dataset.grey_test_dataset, shape=(10*num_of_classes, image_size, image_size, num_channels), name='TEST_DATASET')


    # Training computation.
    logits = nn_model(tf_train_dataset, weights, biases, True)
    loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

    # L2 regularization for the fully connected
    regularizers = (tf.nn.l2_loss(weights['fc1']) +
    tf.nn.l2_loss(biases['fc1']) + tf.nn.l2_loss(weights['fc2']) + tf.nn.l2_loss(biases['fc2']))

    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers
    tf.summary.scalar("loss", loss)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    train_prediction = tf.nn.softmax(nn_model(tf_train_dataset, weights, biases, TRAIN=False))
    # valid_prediction = tf.nn.softmax(nn_model(tf_valid_dataset, weights, biases))
    test_prediction = tf.nn.softmax(nn_model(tf_test_dataset, weights, biases))


    with tf.Session(graph=graph) as session:
        writer = tf.summary.FileWriter(log_location, session.graph)
        merged = tf.summary.merge_all()
        tf.global_variables_initializer().run()
        print("Initialized")

        for step in range(num_steps + 1):
            sys.stdout.write('Training on batch %d of %d\r' % (step+1, num_steps))
            sys.stdout.flush()

            if binary:
                offset = (step * batch_size) % (dataset.bin_train_labels.shape[0] - batch_size)
                #Generate minibatch
                batch_data = dataset.bin_train_dataset[offset: (offset+batch_size), :]
                batch_labels = dataset.bin_train_labels[offset: (offset+batch_size), :]
            else:
                offset = (step * batch_size) % (dataset.grey_train_labels.shape[0] - batch_size)
                # print("shape ",dataset.grey_train_labels.shape[0],"offset ",offset)
                #Generate minibatch
                batch_data = dataset.grey_train_dataset[offset: (offset+batch_size), :]
                batch_labels = dataset.grey_train_labels[offset: (offset+batch_size), :]

            feed_dict = {tf_train_dataset: batch_data, tf_train_labels : batch_labels}
            summary_result, _, l, predictions = session.run([merged, optimizer, loss, train_prediction], feed_dict=feed_dict)
            writer.add_summary(summary_result, step)

            if step % data_showing_step == 0:
                acc_minibatch = accuracy(predictions, batch_labels)
                # acc_val = accuracy(valid_prediction.eval(), dataset.valid_labels)
                acc_val = None
                logger.info('# %03d  Acc Train: %03.2f%%  Loss: %f' % (step, acc_minibatch, l))
        if binary:
            logger.info("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), dataset.bin_test_labels))
            conf = tf.confusion_matrix(labels=dataset.bin_test_labels, predictions=test_prediction.eval(), num_classes=num_classes)
            print(conf)
        else:
            logger.info("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), dataset.grey_test_labels))
            conf = tf.confusion_matrix(labels=dataset.grey_test_labels, predictions=test_prediction.eval(), num_classes=num_classes)
            print(conf)

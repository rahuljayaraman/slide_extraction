import tensorflow as tf
from constants import Constants
import time
import data_utils
import math

IMAGE_PIXELS = Constants.IMAGE_SIZE * Constants.IMAGE_SIZE * 3;
NUM_CLASSES = 2

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 100, 'Batch size')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')

def placeholder_inputs(batch_size):
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_PIXELS))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return images_placeholder, labels_placeholder

def inference(images, hidden1_units, hidden2_units):
  with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                            stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),
                         name='biases')
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
  # Hidden 2
  with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),
                         name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                         name='biases')
    logits = tf.matmul(hidden2, weights) + biases
  return logits

def calculate_loss(logits, labels):
  batch_size = tf.size(labels)
  labels = tf.expand_dims(labels, 1)
  indices = tf.expand_dims(tf.range(0, batch_size), 1)
  concated = tf.concat(1, [indices, labels])
  onehot_labels = tf.sparse_to_dense(
      concated, tf.pack([batch_size, NUM_CLASSES]), 1.0, 0.0)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                          onehot_labels,
                                                          name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss

def training(loss, learning_rate):
  tf.scalar_summary(loss.op.name, loss)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op

def evaluation(logits, labels):
  correct = tf.nn.in_top_k(logits, labels, 1)
  return tf.reduce_sum(tf.cast(correct, tf.int32))

def fill_feed_dict(data_set, images_pl, labels_pl):
  images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict

def run_training():
    with tf.Graph().as_default():
        data_sets = data_utils.read_ann_data_sets()
        images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)

        logits = inference(images_placeholder, FLAGS.hidden1, FLAGS.hidden2)

        loss = calculate_loss(logits, labels_placeholder)

        train_op = training(loss, FLAGS.learning_rate)

        eval_correct = evaluation(logits, labels_placeholder)

        summary_op = tf.merge_all_summaries()

        saver = tf.train.Saver()

        sess = tf.Session()

        init = tf.initialize_all_variables()
        sess.run(init)

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=sess.graph_def)
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            feed_dict = fill_feed_dict(data_sets.train, images_placeholder, labels_placeholder)
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            duration = time.time() - start_time
            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                saver.save(sess, FLAGS.train_dir, global_step=step)
                # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    data_sets.train)
                # Evaluate against the validation set.
                print('Validation Data Eval:')
                do_eval(sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    data_sets.validation)
                # Evaluate against the test set.
                print('Test Data Eval:')
                do_eval(sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    data_sets.test)

def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_set):
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   images_placeholder,
                                   labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = true_count / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))


run_training()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import numpy as np
from six.moves import xrange
import tensorflow as tf
import model
from constants import paths

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', paths.TRAIN_DIR,
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 50000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('resume_training', False,
                            """Resume training?""")
tf.app.flags.DEFINE_integer('run_no', 1,
                            """Allow TB to differentiate runs""")

SUMMARY_DIR = FLAGS.train_dir + '/' + str(FLAGS.run_no)


def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        images, labels = model.distorted_inputs()

        logits = model.inference(images)

        loss = model.loss(logits, labels)

        train_op = model.train(loss, global_step)

        saver = tf.train.Saver(tf.all_variables())

        summary_op = tf.merge_all_summaries()

        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))

        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if FLAGS.resume_training and ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            current_step = int(ckpt.model_checkpoint_path
                               .split('/')[-1].split('-')[-1])
        else:
            current_step = 0
            init = tf.initialize_all_variables()
            sess.run(init)

        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.train.SummaryWriter(SUMMARY_DIR,
                                                graph_def=sess.graph_def)

        for step in xrange(current_step, FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f'
                              '(%.1f examples/sec; %.3f'
                              'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value,
                                     examples_per_sec, sec_per_batch))

            if step % 50 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            if step % 100 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):
    if not FLAGS.resume_training:
        if tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(SUMMARY_DIR)
    if not tf.gfile.Exists(SUMMARY_DIR):
        tf.gfile.MakeDirs(SUMMARY_DIR)
    train()


if __name__ == '__main__':
    tf.app.run()

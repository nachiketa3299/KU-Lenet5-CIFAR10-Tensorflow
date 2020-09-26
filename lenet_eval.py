import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.datasets.cifar10 import load_data

def del_all_flags(_FLAGS):
    flags_dict = _FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        _FLAGS.__delattr__(keys)

(x_train_val, y_train_val), (x_test, y_test) = load_data()

filenames = os.listdir(os.path.join(os.path.curdir, 'runs'))

file = open('./INFOS/log_eval.txt', 'w')
for filename in filenames:
    # Eval Parameters
    del_all_flags(tf.flags.FLAGS)
    tf.reset_default_graph()
    tf.flags.DEFINE_string("checkpoint_dir", f"./runs/{filename}/checkpoints", "Checkpoint directory from training run")
    FLAGS = tf.flags.FLAGS
    # ==================================================
    y_test_one_hot = np.eye(10)[y_test] #one-hot encoding 만들기
    y_test_one_hot = np.squeeze(y_test_one_hot, axis=1)
    x_test = (x_test/127.5) - 1

    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)#가장 validation accuracy가 높은 시점 load
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name, name을 통해 operation 가져오기
            X = graph.get_operation_by_name("X").outputs[0]
            Y = graph.get_operation_by_name("Y").outputs[0]
            keep_prob = graph.get_operation_by_name("keep_prob").outputs[0]
            accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]

            test_accuracy = sess.run(accuracy, feed_dict={X: x_test, Y: y_test_one_hot, keep_prob:1.0})
            file.write(f'{filename}\t{test_accuracy}\n')
            print('- Timestamp        : ', filename)
            print('- Test Max Accuracy: ', test_accuracy)
file.close()

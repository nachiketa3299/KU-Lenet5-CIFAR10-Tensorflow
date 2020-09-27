import tensorflow as tf
import os
import time
import datetime
from tensorflow.keras.datasets.cifar10 import load_data
import data_helpers as dh
from lenet import LeNet
from presets import Preset
import math

def del_all_flags(_FLAGS):
    flags_dict = _FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        _FLAGS.__delattr__(keys)


presets = list(range(1, 21))
for preset in presets:
    p = Preset(preset)
    del_all_flags(tf.flags.FLAGS)
    tf.reset_default_graph()

    tf.flags.DEFINE_integer("SEED", p.SEED, '')
    tf.flags.DEFINE_integer("batch_size", p.batch_size, "Batch Size (default: 64)")
    tf.flags.DEFINE_string("activation_function", p.activation_function, "Activation Function(default: ReLU)")
    if isinstance(p.weight_initialization, str):
        tf.flags.DEFINE_string("weight_initialization", p.weight_initialization, "Weigh Initialization")
    else:
        tf.flags.DEFINE_float("weight_initialization", p.weight_initialization, "Weigh Initialization")
    tf.flags.DEFINE_string("optimizer", p.optimizer, '')
    tf.flags.DEFINE_float("starter_learning_rate", p.starter_learning_rate, "Start Learning Rate (default=0.001)")
    tf.flags.DEFINE_integer("num_epochs", p.num_epochs, "Number of training epochs (default: 200)")
    tf.flags.DEFINE_float("keep_prob", p.keep_prob, "keep probability for dropout (default: 1.0)")
    tf.flags.DEFINE_float("l2_reg_lambda", p.l2_reg_lambda, "L2 regularization lambda (default: 0.0)")
    tf.flags.DEFINE_boolean("data_augmentation", p.data_augmentation, "data augmentation option")
    tf.flags.DEFINE_float("learning_rate_decay_rate", p.learning_rate_decay_rate, "Learning rate decay rate")
    tf.flags.DEFINE_integer("learning_rate_decay_step", p.learning_rate_decay_step, "Learning rate decay step")

    tf.flags.DEFINE_integer("num_classes", 10, "The number of classes (default: 10)")

    # Training parameters
    tf.flags.DEFINE_integer("evaluate_every", 350, "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer("checkpoint_every", 350, "Save model after this many steps (default: 100)")
    tf.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 5)")

    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
    FLAGS = tf.flags.FLAGS

    (x_train_val, y_train_val), (x_test, y_test) = load_data() # training data: 50000, test data: 10000
    x_train, y_train, x_test, y_test, x_val, y_val = dh.shuffle_data(x_train_val, y_train_val, x_test, y_test, FLAGS.num_classes)
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            lenet = LeNet(FLAGS)

            # Define Training procedure
            # * hint learning rate decay를 위한 operation을 통해 감쇠된 learning rate를 optimizer에 적용
            learning_rate = FLAGS.starter_learning_rate # Learning rate decay가 적용되지 않는다면 Starter learning rate값이 그대로 전달됨
            global_step = tf.Variable(0, name="global_step", trainable=False)  # iteration 수
            if FLAGS.learning_rate_decay_rate != 1:
                learning_rate =  learning_rate * FLAGS.learning_rate_decay_rate ** (global_step/FLAGS.learning_rate_decay_step)
            if FLAGS.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) #Optimizer
            elif FLAGS.optimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
            grads_and_vars = optimizer.compute_gradients(lenet.loss) # gradient 계산
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step) # back-propagation

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print(f"Writing to {out_dir}\n")

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", lenet.loss)
            acc_summary = tf.summary.scalar("accuracy", lenet.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            sess.run(tf.global_variables_initializer()) # 모든 가중치 초기화

            def train_step(_x_batch, _y_batch):
                feed_dict = {lenet.X: _x_batch, lenet.Y: _y_batch, lenet.keep_prob: FLAGS.keep_prob}
                # * hint learning rate decay operation 실행
                learning_rate_dc = FLAGS.starter_learning_rate
                if FLAGS.learning_rate_decay_rate == 1:
                    _, step, summaries, loss, accuracy = sess.run([train_op, global_step, train_summary_op, lenet.loss, lenet.accuracy], feed_dict)
                else:
                    _, step, summaries, loss, accuracy, learning_rate_dc = sess.run([train_op, global_step, train_summary_op, lenet.loss, lenet.accuracy, learning_rate], feed_dict)

                time_str = datetime.datetime.now().strftime("%m/%d %H:%M:%S")
                prog = FLAGS.num_epochs * math.ceil(45000/FLAGS.batch_size)
                print(f"Preset={preset}  [{time_str}]  Step=({format(step, '05')}/{format(prog, '05')})  Loss={format(loss, '<8g')}  Acc={format(accuracy, '<9g')}  LR={format(learning_rate_dc, '<21g')}")
                train_summary_writer.add_summary(summaries, step)

            def dev_step(_x_batch, _y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {lenet.X: _x_batch, lenet.Y: _y_batch, lenet.keep_prob: 1.0}
                step, summaries, loss, accuracy = sess.run([global_step, dev_summary_op, lenet.loss, lenet.accuracy], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print(f"[{time_str}]  Step={format(step, '05')}  Loss={format(loss, '<8g')}  Acc={format(accuracy, '<9g')}")
                if writer:
                    writer.add_summary(summaries, step)
                return accuracy

            # Generate batches
            if FLAGS.data_augmentation: # data augmentation 적용시
                batches = dh.batch_iter_aug(x_train, y_train, FLAGS.batch_size, FLAGS.num_epochs)
            else:
                batches = dh.batch_iter(x_train, y_train, FLAGS.batch_size, FLAGS.num_epochs)

            # Training loop. For each batch...
            max = 0
            start_time = time.time()


            for batch in batches: # len(batches) = (45000/batch size) * epoch 수
                x_batch, y_batch = zip(*batch) # batch size 단위로 input과 정답 리턴, e.g., (128, 32, 32, 3), (128, 10),
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0: # 특정 iteration 마다
                    print("\n>> Evaluation:")
                    accuracy = dev_step(x_val, y_val, writer=dev_summary_writer) # validation accuracy 확인
                    print("")
                    if accuracy > max: # validation accuracy가 경신될 때
                        max = accuracy
                        early_stopping_epoch = current_step
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step) # best accuracy에 도달할 때만 모델을 저장함으로써 early stopping
                        print(f"> Saved model checkpoint to {path}\n")
            training_time = (time.time() - start_time) / 60
            print(f'>> Training time: {training_time}')
            filename ='./INFOS/train_result.txt'
            with open(filename, 'a') as file:
                if os.path.getsize(filename) == 0:
                    file.write("batch_size\tactivation_function\tweight__initialization\toptimizer\tstarter_learning_rate\tnum_epochs\tkeep_prob\tl2_reg_lambda\tdata_augmentation\tlearning_rate_decay_rate\tlearning_rate_decay_step\ttraining_time\tearly_stopping_epoch\taccuracy\ttimstamp\n")
                file.write(f"{p.batch_size}\t{p.activation_function}\t{p.weight_initialization}\t{p.optimizer}\t{p.starter_learning_rate}\t{p.num_epochs}\t{p.keep_prob}\t{p.l2_reg_lambda}\t{p.data_augmentation}\t{p.learning_rate_decay_rate}\t{p.learning_rate_decay_step}\t{training_time}\t{early_stopping_epoch}\t{accuracy}\t{timestamp}\n")




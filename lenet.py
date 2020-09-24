import tensorflow as tf
import numpy as np
import math

# lenet = LeNet(FLAG)로 초기화 됨.
'''
- learning rate (default = 0.1)
- learning rate decay rate (default = 0.1)
- L2 Regularization lambda (default = 0.1)
- dropout: keep_prob (default = 1.0)
- number of classes (default = 10)

- batch_size (default = 64)
- num_epoch (default = 200)
- evaluate_every: evaluate model on dev set after this many steps (default = 100)
- num_checkpoints: number of checkpoints to store (default = 5)
- data_augmentation (default = True)

- allow soft placement = True
- log device placement = False
'''
class LeNet:
    def __init__(self, config):
        # 정답 Label 갯수 - 10개 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
        self._num_classes = config.num_classes
        # Weight Decay 위한 Lambda 값
        self._l2_reg_lambda = config.l2_reg_lambda

        # (32, 32, 3)의 RGB 3채널 갖는 Input 텐서 선언
        self.X = tf.placeholder(tf.float32, [config.batch_size, 32, 32, 3], name="X")
        # 정답이 들어올 자리를 원 핫 인코딩 형태로 선언
        self.Y = tf.placeholder(tf.float32, [config.batch_size, self._num_classes], name="Y")
        # Dropout 적용시 살릴 확률
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        #  TODO : LeNet5 모델 생성
        # Input으로 들어올 데이터 사이즈 (32, 32, 3)
        # LAYER 1: (32, 32, 3) -> (28 * 28 * 6) | filter = (5 * 5 * 3) * 6
        # * hint he initialization: stddev = sqrt(2/n), filter에서 n 값은 이전 레이어의 노드 수.
        self.conv_f1 = tf.Variable(tf.random_normal([5, 5, 3, 6], stddev=math.sqrt(2 / (32 * 32 * 3))), name='conv_f1')
        self.C1 = tf.nn.conv2d(self.X, self.conv_f1, strides=[1, 1, 1, 1], padding='SAME')

        # LAYER 2: (28 * 28 * 6) -> ReLu -> (28 * 28 * 6) -> max_pooling(2*2, stride=2) -> (14 * 14 * 6)
        self.S2 = tf.nn.max_pool(tf.nn.relu(self.C1), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # LAYER 3: (14 * 14 * 6) -> (10 * 10 * 16) | filter = (5 * 5 * 6) * 16
        self.conv_f2 = tf.Variable(tf.random_normal([5, 5, 6, 16], stddev=math.sqrt(2 / (14 * 14 * 6))), name='conv_f2')
        self.C3 = tf.nn.conv2d(self.S2, self.conv_f2, strides=[1, 1, 1, 1], padding='SAME')

        # LAYER 4: (10 * 10 * 16) -> ReLu -> (10 * 10 * 16) -> max_pooling(2*2, stride=2) -> (5, 5, 16)
        self.S4 = tf.nn.max_pool(tf.nn.relu(self.C3), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # 평탄화 -> (5 * 5 *16)

        # LAYER 5(FC1): (5, 5, 16) -> (5 * 5 * 16 = 400) | (batch_size, 400) * (400, 120) -> (batch_size, 120)
        # FC1 추가 (5 * 5 * 16, 120) -> (120)
        self.FC1 = tf.reshape(self.S4, [config.batch_size, 5, 5, 16], name="FC1")
        self.FCW1 = tf.get_variable("FCW1", shape=[5 * 5 * 16, 120])
        self.FCB1 = tf.Variable(tf.random_normal([120]), name="FCB1")

        # LAYER 6(FC2): (batch_size, 120) * (120, 84) -> (batch_size, 84)
        self.FC2 = tf.nn.dropout(tf.matmul(self.FC1, self.FCW1) + self.FCB1, keep_prob=config.keep_prob)
        self.FCW2 = tf.get_variable("FCW2", shape=[120, 84])
        self.FCB2 = tf.Variable(tf.random_normal([84]), name="FCB2")

        # LAYER 7(Softmax): (batch_size, 84) * (84, 10) -> softmax -> (batch_size, 10)
        self.FC3 = tf.nn.dropout(tf.matmul(self.FC2, self.FCW2) + self.FCB2, keep_prob=config.keep_prob)
        self.FCW3 = tf.get_variable("FCW3", shape=[84, 10])
        self.FCB3 = tf.Variable(tf.random_normal([10]), name="FCB3")
        hypothesis = tf.nn.softmax(tf.nn.xw_plus_b(self.FC3, self.FCW3, self.FCB3, name="hypothesis"))

        with tf.variable_scope('logit'):
          self.predictions = tf.argmax(hypothesis, 1, name="predictions") # logit/hypothesis

        with tf.variable_scope('loss'):
          costs = []
          for var in tf.trainable_variables():
              costs.append(tf.nn.l2_loss(var)) # 모든 가중치들의 l2_loss 누적
          l2_loss = tf.add_n(costs)
          xent = tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=self.Y)
          self.loss = tf.reduce_mean(xent, name='xent') + self._l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

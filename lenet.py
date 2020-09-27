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
        tf.set_random_seed(config.SEED)

        self._num_classes = config.num_classes
        # Weight Decay 위한 Lambda 값
        self._l2_reg_lambda = config.l2_reg_lambda

        self.X = tf.placeholder(tf.float32, [None, 32, 32, 3], name="X")
        self.Y = tf.placeholder(tf.float32, [None, self._num_classes], name="Y")
        # Dropout 적용시 살릴 확률
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        # Weight Initialization
        # self.initializer = None
        # stddev_fun = None
        # if config.weight_initialization == 'xe':
        #     stddev_fun = lambda n : math.sqrt(1 / n)
        #
        # elif config.weight_initialization == 'he':
        #     stddev_fun = lambda n : math.sqrt(2 / n) # n is filter size * filter size * out channel
        # else:
        #     self.initializer = tf.random_normal_initializer(stddev=config.weight_initialization)
        #
        # if config.weight_initialization == 'xe' or config.weight_initialization == 'he':
        #     self.initializer = []
        #     self.initializer.append(tf.random_normal_initializer(stddev=stddev_fun(32 * 32 * 3)))
        #     self.initializer.append(tf.random_normal_initializer(stddev=stddev_fun(14 * 14 * 6)))
        #     self.initializer.append(tf.random_normal_initializer(stddev=stddev_fun(5 * 5 * 16)))
        #     self.initializer.append(tf.random_normal_initializer(stddev=stddev_fun(120)))
        #     self.initializer.append(tf.random_normal_initializer(stddev=stddev_fun(84)))
        # else:
        #     self.conv_f1 = tf.get_variable(name='conv_f1', shape=[5, 5, 3, 6], initializer=self.initializer)
        #     self.conv_f2 = tf.get_variable(name='conv_f2', shape=[5, 5, 6, 16], initializer=self.initializer)
        #     self.FCW1 = tf.get_variable(name="FCW1", shape=[5 * 5 * 16, 120], initializer=self.initializer)
        #     self.FCW2 = tf.get_variable(name="FCW2", shape=[120, 84], initializer=self.initializer)
        #     self.FCW3 = tf.get_variable(name="FCW3", shape=[84, 10], initializer=self.initializer)
        init = None
        if config.weight_initialization == 'xe':
            init = tf.contrib.layers.xavier_initializer()
        elif config.weight_initialization == 'he':
            init = tf.contrib.layers.variance_scaling_initializer()
        else:
            init = tf.random_normal_initializer(stddev=config.weight_initialization)

        self.conv_f1 = tf.get_variable(name='conv_f1', shape=[5, 5, 3, 6], initializer=init)
        self.conv_f2 = tf.get_variable(name='conv_f2', shape=[5, 5, 6, 16], initializer=init)
        self.FCW1 = tf.get_variable(name="FCW1", shape=[5 * 5 * 16, 120], initializer=init)
        self.FCW2 = tf.get_variable(name="FCW2", shape=[120, 84], initializer=init)
        self.FCW3 = tf.get_variable(name="FCW3", shape=[84, 10], initializer=init)
        self.FCB1 = tf.Variable(tf.random_normal([120]), name="FCB1")
        self.FCB2 = tf.Variable(tf.random_normal([84]), name="FCB2")
        self.FCB3 = tf.Variable(tf.random_normal([10]), name="FCB3")

        #  TODO : LeNet5 모델 생성
        # Input으로 들어올 데이터 사이즈 (32, 32, 3)
        # LAYER 1: (32, 32, 3) -> (28 * 28 * 6) | filter = (5 * 5 * 3) * 6
        # * hint he initialization: stddev = sqrt(2/n), filter에서 n 값은 이전 레이어의 노드 수.
        # self.conv_f1 = tf.Variable(tf.random_normal([5, 5, 3, 6], stddev=math.sqrt(2 / (32 * 32 * 3))), name='conv_f1')
        self.C1 = tf.nn.conv2d(self.X, self.conv_f1, strides=[1, 1, 1, 1], padding='VALID', name='C1')
        # Select Activation Function
        if config.activation_function == 'relu':
            self.C1 = tf.nn.relu(self.C1)
        elif config.activation_function == 'sigmoid':
            self.C1 = tf.nn.sigmoid(self.C1)

        # LAYER 2: (28 * 28 * 6) -> ReLU -> (28 * 28 * 6) -> max_pooling(2*2, stride=2) -> (14 * 14 * 6)
        self.S2 = tf.nn.max_pool(self.C1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='S2')

        # LAYER 3: (14 * 14 * 6) -> (10 * 10 * 16) | filter = (5 * 5 * 6) * 16
        # self.conv_f2 = tf.Variable(tf.random_normal([5, 5, 6, 16], stddev=math.sqrt(2 / (14 * 14 * 6))), name='conv_f2')
        self.C3 = tf.nn.conv2d(self.S2, self.conv_f2, strides=[1, 1, 1, 1], padding='VALID', name='C3')
        # Select Activation Function
        if config.activation_function == 'relu':
            self.C3 = tf.nn.relu(self.C3)
        elif config.activation_function == 'sigmoid':
            self.C3 = tf.nn.sigmoid(self.C3)

        # LAYER 4: (10 * 10 * 16) -> ReLU -> (10 * 10 * 16) -> max_pooling(2*2, stride=2) -> (5, 5, 16)
        self.S4 = tf.nn.max_pool(self.C3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='S4')

        # LAYER 5(FC1): (5, 5, 16) -> (5 * 5 * 16 = 400) | (batch_size, 400) * (400, 120) -> (batch_size, 120)
        # self.FC1 = tf.reshape(self.S4, [config.batch_size, 5, 5, 16], name="FC1")
        self.FC1 = tf.compat.v1.layers.flatten(self.S4, name="FC1")
        # LAYER 6(FC2): (batch_size, 120) * (120, 84) -> (batch_size, 84)
        self.FC1 = tf.matmul(self.FC1, self.FCW1) + self.FCB1
        # Select Activation Function
        if config.activation_function == 'relu':
            self.FC1 = tf.nn.relu(self.FC1)
        elif config.activation_function == 'sigmoid':
            self.FC1 = tf.nn.sigmoid(self.FC1)
        self.FC2 = tf.nn.dropout(self.FC1, keep_prob=config.keep_prob, name="FC2")

        # LAYER 7(Softmax): (batch_size, 84) * (84, 10) -> softmax -> (batch_size, 10)
        self.FC2 = tf.matmul(self.FC2, self.FCW2) + self.FCB2
        # Select Activation Function
        if config.activation_function == 'relu':
            self.FC2 = tf.nn.relu(self.FC2)
        elif config.activation_function == 'sigmoid':
            self.FC2 = tf.nn.sigmoid(self.FC2)
        self.FC3 = tf.nn.dropout(self.FC2, keep_prob=config.keep_prob, name="FC3")

        ############# for debug ##############
        # print('- X      ', self.X.shape)
        # print('- conv_f1', self.conv_f1.shape)
        # print('- C1     ', self.C1.shape)
        # print('- S2     ', self.S2.shape)
        # print('- conv_f2', self.conv_f2.shape)
        # print('- C3     ', self.C3.shape)
        # print('- S4     ', self.S4.shape)
        # print('- FC1    ', self.FC1.shape)
        # print('- FCW1   ', self.FCW1.shape)
        # print('- FCB1   ', self.FCB1.shape)
        # print('- FC2    ', self.FC2.shape)
        # print('- FCW2   ', self.FCW2.shape)
        # print('- FCB2   ', self.FCB2.shape)
        # print('- FC3    ', self.FC3.shape)
        # print('- FCW3   ', self.FCW3.shape)
        # print('- FCB3   ', self.FCB3.shape)
        ######################################
        hypothesis = tf.nn.softmax(tf.nn.xw_plus_b(self.FC3, self.FCW3, self.FCB3, name="hypothesis"))

        with tf.variable_scope('logit'):
          self.predictions = tf.argmax(hypothesis, 1, name="predictions") # logit/hypothesis

        with tf.variable_scope('loss'):
          costs = []
          for var in tf.trainable_variables(): # Returns all variables created with trainable=True
              # costs = [conv_f1, conv_f2, FCW1, FCW2, FCW3]
              costs.append(tf.nn.l2_loss(var)) # 모든 가중치들의 l2_loss 누적
          l2_loss = tf.add_n(costs)
          xent = tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=self.Y)
          self.loss = tf.reduce_mean(xent, name='xent') + self._l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

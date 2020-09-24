class Preset:
    def __init__(self, preset_num=1):
        if preset_num == 1:
            self.batch_size = 128
            self.activation_function = 'ReLu'
            self.weight_initialization = 0.01
            self.optimizer = 'adam'
            self.learning_rate = 0.001
            self.epoch = 200
            self.dropout = 0.1
            self.weight_decay = 0
            self.data_augmentation = False
            self.lr_decay = 0.1
        elif preset_num == 2:
            self.batch_size = 128
            self.activation_function = 'ReLu'
            self.weight_initialization = 0.01
            self.optimizer = 'adam'
            self.learning_rate = 0.001
            self.epoch = 200
            self.dropout = 0.1
            self.weight_decay = 0.0001
            self.data_augmentation = False
            self.lr_decay = 0.1
        elif preset_num == 3:
            self.batch_size = 128
            self.activation_function = 'ReLu'
            self.weight_initialization = 0.01
            self.optimizer = 'adam'
            self.learning_rate = 0.001
            self.epoch = 200
            self.dropout = 0.1
            self.weight_decay = 0.0001
            self.data_augmentation = True
            self.lr_decay = 0.1
        elif preset_num == 4:
            self.batch_size = 128
            self.activation_function = 'ReLu'
            self.weight_initialization = 0.01
            self.optimizer = 'adam'
            self.learning_rate = 0.001
            self.epoch = 200
            self.dropout = 0.1
            self.weight_decay = 0.0001
            self.data_augmentation = True
            self.lr_decay = True
    def setParameters(self, _batch_size, _activation_function, _weight_initialization, _optimizer, _learning_rate, _epoch, _dropout, _weight_decay, _data_augmentation, _lr_decay):
        self.batch_size = _batch_size
        self.activation_function = _activation_function
        self.weight_initialization =  _weight_initialization
        self.optimizer = _optimizer
        self.learning_rate =  _learning_rate
        self.epoch = _epoch
        self.dropout = _dropout
        self.weight_decay = _weight_decay
        self.data_augmentation = _data_augmentation
        self.lr_decay = _lr_decay





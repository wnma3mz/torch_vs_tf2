import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten


class CNN(Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = Conv2D(32, 3, activation="relu")
        self.flatten = Flatten()
        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(10, activation="softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


# 专家
class MLP(Model):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(10, activation="softmax")

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


# 初学者
class MLPSeq:
    def build(self):
        # 将模型的各层堆叠起来，以搭建 tf.keras.models.Sequential 模型，为训练选择优化器和损失函数
        return tf.keras.models.Sequential(
            [Flatten(), Dense(128, activation="relu"), Dense(10, activation="softmax")]
        )


class CNNSeq:
    def build(self):
        return tf.keras.models.Sequential(
            [
                Conv2D(32, 3, activation="relu"),
                Flatten(),
                Dense(100, activation="relu"),
                Dense(10, activation="softmax"),
            ]
        )

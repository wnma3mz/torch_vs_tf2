import numpy as np
import tensorflow as tf
from tqdm import tqdm


# 显示训练/测试过程
def show_f(fn):
    def wrapper(self, loader):
        if self.display == True:
            with tqdm(loader, ncols=80, postfix="loss: *.****; acc: *.**") as t:
                return fn(self, t)
        return fn(self, loader)

    return wrapper


class Trainer:
    def __init__(self, model, optimizer, criterion, display=True):
        self.model = model
        self.display = display
        self.criterion = criterion
        self.optimizer = optimizer

        self.iter_loss_f = tf.keras.metrics.Mean()
        self.iter_accuracy_f = tf.keras.metrics.SparseCategoricalAccuracy()

    @tf.function
    def batch(self, data, target):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(data, training=self.is_train)
            loss = self.criterion(target, predictions)
        if self.is_train:
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )

        return self.iter_loss_f(loss), self.iter_accuracy_f(target, predictions)

    @show_f
    def _iteration(self, loader):
        """模型训练/测试的入口函数, 控制输出显示

        Args:
            data_loader : torch.utils.data
                          数据集

        Returns:
            float :
                    每个epoch的loss取平均

            float :
                    每个epoch的accuracy取平均
        """

        loop_loss, loop_accuracy = [], []
        for data, target in loader:
            iter_loss, iter_acc = self.batch(data, target)
            loop_accuracy.append(iter_acc)
            loop_loss.append(iter_loss)

            if self.display:
                loader.postfix = "loss: {:.4f}; acc: {:.2f}".format(iter_loss, iter_acc)
        return np.mean(loop_loss), np.mean(loop_accuracy)

    def train(self, data_loader, epochs=1):
        self.is_train = True
        epoch_loss, epoch_accuracy = [], []
        for _ in range(1, epochs + 1):
            loss, accuracy = self._iteration(data_loader)
            epoch_loss.append(loss)
            epoch_accuracy.append(accuracy)
        return np.mean(epoch_loss), np.mean(epoch_accuracy)

    def test(self, data_loader):
        self.is_train = False
        loss, accuracy = self._iteration(data_loader)
        return loss, accuracy

    def save(self, fpath):
        """保存模型
        Args:
            fpath :  string
                     模型保存的路径
        """
        # self.model.save(fpath)
        self.model.save_weights(fpath)

    def restore(self, fpath):
        """恢复模型
        Args:
            fpath :  string
                     模型保存的路径
        """
        # self.model = tf.keras.models.load_model(fpath)
        self.model.load_weights(fpath)


class TrainerSeq:
    def __init__(self, model, optimizer, criterion):
        self.model = model

        self.model.compile(
            optimizer=optimizer,
            loss=criterion,
            metrics=["accuracy"],
        )
        self.callbacks = []

    def train(self, data_loader, epochs=1, verbose=1):
        """
        verbose = 0: 不显示输出
        verbose = 1: 显示输出
        verbose = 2: 每个epoch一行输出
        """
        fpath = "./ckpt/cp-{epoch:04d}.ckpt"
        self.save(fpath)
        history_data = self.model.fit(
            data_loader, epochs=epochs, verbose=verbose, callbacks=self.callbacks
        )
        self.loss, self.accuracy = (
            history_data.history["loss"],
            history_data.history["accuracy"],
        )
        return self.loss[-1], self.accuracy[-1]

    def test(self, data_loader, verbose=2):
        loss, accuracy = self.model.evaluate(data_loader, verbose=verbose)
        return loss, accuracy

    def save(self, fpath):
        # period 每隔5epoch保存一次
        self.callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                fpath, save_weights_only=True, verbose=1, period=5
            )
        )

        # self.callbacks.append(
        #     tf.keras.callbacks.ModelCheckpoint(
        #         fpath, save_weights_only=False, verbose=1, period=5
        #     )
        # )

    def restore(self, fpath):
        # self.model = tf.keras.models.load_model(fpath)
        self.model.load_weights(fpath)
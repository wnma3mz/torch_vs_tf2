import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10, cifar100, mnist


def augment(x, y, mean=0.5, std=0.5, aug=False):
    # 转换数据类型
    x, y = x.astype(np.float32), y.astype(np.float32)
    # 数据增强
    if aug:
        data_augmentation = tf.keras.Sequential(
            [
                layers.experimental.preprocessing.RandomCrop(32, 32),
                # layers.ZeroPadding2D(padding=2),  # 上下左右各2，不等效于RandomCrop(32, padding=4)
                # "horizontal", "vertical", or "horizontal_and_vertical"
                layers.experimental.preprocessing.RandomFlip("horizontal"),
            ]
        )
        x = data_augmentation(x)

    # 归一化
    x = layers.experimental.preprocessing.Normalization(mean=mean, variance=std)(x)
    # x = x / 255.0
    return x, y


def get_datasets(dataset_name, batch_size):

    mean, std = 0.5, 0.5
    if dataset_name == "mnist":
        mean, std = 0.5, 0.5
        # 载入并准备好 MNIST 数据集
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # MNIST处理
        reshape_f = lambda x: x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
        x_train = reshape_f(x_train)
        x_test = reshape_f(x_test)

    elif dataset_name == "cifar10":
        # mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif dataset_name == "cifar100":
        # mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        # one of "fine", "coarse". If it is "fine" the category labels are the fine-grained labels, if it is "coarse" the output labels are the coarse-grained superclasses.
        (x_train, y_train), (x_test, y_test) = cifar100.load_data("fine")
    else:
        raise NotImplementedError("This dataset is not currently supported")

    if dataset_name == "mnist":
        x_train, y_train = augment(x_train, y_train, mean=mean, std=std)
    else:
        x_train, y_train = augment(x_train, y_train, mean=mean, std=std, aug=True)
    x_test, y_test = augment(x_test, y_test, mean=mean, std=std)

    # 随机切分，并定义batch_size
    trainloader = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(batch_size * 10)
        .batch(batch_size)
    )

    testloader = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    return trainloader, testloader

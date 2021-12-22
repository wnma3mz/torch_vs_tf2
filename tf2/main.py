import time

import tensorflow as tf

from datasets import get_datasets
from models import CNN, MLP, CNNSeq, MLPSeq
from Trainer import Trainer, TrainerSeq

if __name__ == "__main__":
    batch_size = 128
    trainloader, testloader = get_datasets("cifar10", batch_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # model = MLP()
    # model = CNN()
    # trainer = Trainer(model, optimizer=optimizer, criterion=criterion)

    mlp_seq = MLPSeq()
    model_seq = mlp_seq.build()

    cnn_seq = CNNSeq()
    model_seq = cnn_seq.build()
    trainer = TrainerSeq(model_seq, optimizer=optimizer, criterion=criterion)

    # 模型可视化
    # tf.keras.utils.plot_model(
    #     model,
    #     to_file='model.png',
    #     show_shapes=True,
    #     show_layer_names=True,
    #     rankdir='TB',
    #     expand_nested=True,
    #     dpi=96
    # )

    
    s1 = time.time()
    # 训练并验证模型
    trainer.train(trainloader, epochs=5)
    e1 = time.time()
    print("训练耗时: {}".format(e1 - s1))

    s2 = time.time()
    trainer.test(testloader)
    e2 = time.time()
    print("测试耗时: {}".format(e2 - s2))
